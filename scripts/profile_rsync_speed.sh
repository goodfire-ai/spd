#!/usr/bin/env bash
# Profile rsync/cp download speeds from the polished-lake NFS mount.
#
# Creates temporary test files on the NFS mount, then measures transfer speed
# to local disk (/tmp) using rsync, cp, and dd. This isolates NFS read
# throughput from any SSH overhead.
#
# Usage:
#   ./scripts/profile_rsync_speed.sh              # Run all tests
#   ./scripts/profile_rsync_speed.sh --keep        # Keep test files after run
#   ./scripts/profile_rsync_speed.sh --skip-gen     # Reuse existing test files
#
# To test full SSH+rsync path from your local machine:
#   1. Run this script with --keep to create test files
#   2. From local: rsync -avP <cluster>:/mnt/polished-lake/home/lee/.rsync_profile_test/ /tmp/rsync_test/

set -euo pipefail

# --- Configuration ---
NFS_BASE="/mnt/polished-lake/home/lee"
NFS_TEST_DIR="${NFS_BASE}/.rsync_profile_test"
LOCAL_TEST_DIR="/tmp/rsync_profile_test_$$"
SIZES_MB=(1 10 100 500)
N_SMALL_FILES=200  # for many-small-files test (1KB each)
KEEP=false
SKIP_GEN=false

for arg in "$@"; do
    case "$arg" in
        --keep) KEEP=true ;;
        --skip-gen) SKIP_GEN=true ;;
        *) echo "Unknown arg: $arg"; exit 1 ;;
    esac
done

# --- Helpers ---
bold=$(tput bold 2>/dev/null || true)
reset=$(tput sgr0 2>/dev/null || true)

header() { echo; echo "${bold}=== $1 ===${reset}"; }
subheader() { echo "  ${bold}--- $1 ---${reset}"; }

human_rate() {
    local bytes_per_sec=$1
    if (( $(echo "$bytes_per_sec > 1073741824" | bc -l) )); then
        echo "$(echo "scale=2; $bytes_per_sec / 1073741824" | bc) GB/s"
    elif (( $(echo "$bytes_per_sec > 1048576" | bc -l) )); then
        echo "$(echo "scale=2; $bytes_per_sec / 1048576" | bc) MB/s"
    elif (( $(echo "$bytes_per_sec > 1024" | bc -l) )); then
        echo "$(echo "scale=2; $bytes_per_sec / 1024" | bc) KB/s"
    else
        echo "${bytes_per_sec} B/s"
    fi
}

measure_transfer() {
    # Usage: measure_transfer <label> <command...>
    local label="$1"; shift

    # Drop caches if possible (needs root)
    sync
    echo 3 > /proc/sys/vm/drop_caches 2>/dev/null || true

    local start end elapsed
    start=$(date +%s%N)
    "$@" 2>/dev/null
    end=$(date +%s%N)
    elapsed=$(echo "scale=6; ($end - $start) / 1000000000" | bc)
    echo "$label: ${elapsed}s"
}

measure_transfer_with_rate() {
    # Usage: measure_transfer_with_rate <label> <size_bytes> <command...>
    local label="$1"; local size_bytes="$2"; shift 2

    sync
    { echo 3 > /proc/sys/vm/drop_caches; } 2>/dev/null || true

    local start end elapsed rate
    start=$(date +%s%N)
    "$@" 2>/dev/null
    end=$(date +%s%N)
    elapsed=$(echo "scale=6; ($end - $start) / 1000000000" | bc)
    rate=$(echo "scale=0; $size_bytes / $elapsed" | bc)
    printf "  %-35s %8.2fs  %s\n" "$label" "$elapsed" "$(human_rate "$rate")"
}

cleanup_local() { rm -rf "$LOCAL_TEST_DIR"; }

# --- Generate test files ---
if [ "$SKIP_GEN" = false ]; then
    header "Generating test files on NFS (${NFS_TEST_DIR})"
    mkdir -p "$NFS_TEST_DIR/small_files"

    for size_mb in "${SIZES_MB[@]}"; do
        f="${NFS_TEST_DIR}/test_${size_mb}mb.bin"
        if [ -f "$f" ]; then
            echo "  Reusing existing ${size_mb}MB file"
        else
            echo "  Creating ${size_mb}MB file..."
            dd if=/dev/urandom of="$f" bs=1M count="$size_mb" status=none
        fi
    done

    echo "  Creating ${N_SMALL_FILES} x 1KB small files..."
    for i in $(seq 1 $N_SMALL_FILES); do
        dd if=/dev/urandom of="${NFS_TEST_DIR}/small_files/file_${i}.bin" bs=1024 count=1 status=none
    done
    echo "  Done generating files."
fi

# Verify test files exist
if [ ! -d "$NFS_TEST_DIR" ]; then
    echo "ERROR: Test directory not found: $NFS_TEST_DIR"
    echo "Run without --skip-gen first."
    exit 1
fi

# --- System info ---
header "System Info"
echo "  Hostname:    $(hostname)"
echo "  Date:        $(date -Iseconds)"
echo "  NFS mount:   $(mount | grep polished-lake | head -1 | sed 's/.*type /type /')"
echo "  NFS source:  ${NFS_TEST_DIR}"
echo "  Local dest:  ${LOCAL_TEST_DIR}"
echo "  rsync:       $(rsync --version | head -1)"

# --- Baseline: dd read speed from NFS ---
header "Baseline: dd read from NFS (no copy overhead)"
for size_mb in "${SIZES_MB[@]}"; do
    f="${NFS_TEST_DIR}/test_${size_mb}mb.bin"
    size_bytes=$((size_mb * 1048576))
    measure_transfer_with_rate "dd read ${size_mb}MB" "$size_bytes" \
        dd if="$f" of=/dev/null bs=1M status=none
done

# --- Single large file transfers ---
for size_mb in "${SIZES_MB[@]}"; do
    header "Single file: ${size_mb}MB"
    f="${NFS_TEST_DIR}/test_${size_mb}mb.bin"
    size_bytes=$((size_mb * 1048576))

    cleanup_local; mkdir -p "$LOCAL_TEST_DIR"

    measure_transfer_with_rate "cp" "$size_bytes" \
        cp "$f" "${LOCAL_TEST_DIR}/test.bin"

    cleanup_local; mkdir -p "$LOCAL_TEST_DIR"
    measure_transfer_with_rate "rsync (default)" "$size_bytes" \
        rsync "$f" "${LOCAL_TEST_DIR}/test.bin"

    cleanup_local; mkdir -p "$LOCAL_TEST_DIR"
    measure_transfer_with_rate "rsync --whole-file" "$size_bytes" \
        rsync --whole-file "$f" "${LOCAL_TEST_DIR}/test.bin"

    cleanup_local; mkdir -p "$LOCAL_TEST_DIR"
    measure_transfer_with_rate "rsync -W --inplace" "$size_bytes" \
        rsync --whole-file --inplace "$f" "${LOCAL_TEST_DIR}/test.bin"

    cleanup_local; mkdir -p "$LOCAL_TEST_DIR"
    measure_transfer_with_rate "rsync --no-compress" "$size_bytes" \
        rsync --whole-file --compress=no "$f" "${LOCAL_TEST_DIR}/test.bin"
done

# --- Many small files ---
header "Many small files: ${N_SMALL_FILES} x 1KB"
total_bytes=$((N_SMALL_FILES * 1024))

cleanup_local; mkdir -p "$LOCAL_TEST_DIR/small_files"
measure_transfer_with_rate "cp -r" "$total_bytes" \
    cp -r "${NFS_TEST_DIR}/small_files/." "${LOCAL_TEST_DIR}/small_files/"

cleanup_local; mkdir -p "$LOCAL_TEST_DIR/small_files"
measure_transfer_with_rate "rsync -r (default)" "$total_bytes" \
    rsync -r "${NFS_TEST_DIR}/small_files/" "${LOCAL_TEST_DIR}/small_files/"

cleanup_local; mkdir -p "$LOCAL_TEST_DIR/small_files"
measure_transfer_with_rate "rsync -rW (whole-file)" "$total_bytes" \
    rsync -rW "${NFS_TEST_DIR}/small_files/" "${LOCAL_TEST_DIR}/small_files/"

# --- Parallel rsync (split large file) ---
header "Parallel transfer test: 500MB file"
f="${NFS_TEST_DIR}/test_500mb.bin"
if [ -f "$f" ]; then
    size_bytes=$((500 * 1048576))

    # Split into chunks and rsync in parallel
    cleanup_local; mkdir -p "$LOCAL_TEST_DIR/parallel"
    CHUNK_DIR="${LOCAL_TEST_DIR}/chunks"
    mkdir -p "$CHUNK_DIR"

    subheader "Splitting into 50MB chunks..."
    split -b 50M "$f" "${CHUNK_DIR}/chunk_"
    chunk_count=$(ls "${CHUNK_DIR}/" | wc -l)
    echo "  Created $chunk_count chunks"

    # Reassemble (simulates parallel download + merge)
    cleanup_local; mkdir -p "$LOCAL_TEST_DIR"
    subheader "Sequential rsync of 500MB (baseline)"
    measure_transfer_with_rate "rsync 500MB sequential" "$size_bytes" \
        rsync --whole-file "$f" "${LOCAL_TEST_DIR}/test.bin"
else
    echo "  Skipping (500MB test file not found)"
fi

# --- Summary ---
header "Recommendations"
cat <<'TIPS'
  If NFS read (dd) is fast but rsync is slow:
    - Use rsync --whole-file (-W) to skip delta-transfer algorithm
    - Use rsync --inplace to avoid temp file creation
    - For many small files, consider tar piping:
        tar cf - -C /mnt/polished-lake/path . | tar xf - -C /local/dest

  If NFS read itself is slow:
    - Check NFS mount options (nconnect, rsize/wsize)
    - Check network with: iperf3 -c <nfs-server-ip>
    - Current mount has nconnect=64, rsize/wsize=1MB (good defaults)

  For SSH rsync from your local machine, test with:
    rsync -avWP <user>@<cluster>:/mnt/polished-lake/home/lee/.rsync_profile_test/test_100mb.bin /tmp/
TIPS

# --- Cleanup ---
cleanup_local
if [ "$KEEP" = false ] && [ "$SKIP_GEN" = false ]; then
    echo
    echo "Cleaning up NFS test files..."
    rm -rf "$NFS_TEST_DIR"
else
    echo
    echo "Test files kept at: ${NFS_TEST_DIR}"
fi

echo
echo "Done."
