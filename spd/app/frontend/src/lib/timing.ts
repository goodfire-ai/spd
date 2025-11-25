export function logTiming(event: string, durationMs: number, extra: Record<string, string | number> = {}): void {
    const extraStr = Object.keys(extra).length > 0 ? ` ${JSON.stringify(extra)}` : "";
    console.log(`[timing] ${event}: ${durationMs.toFixed(2)}ms${extraStr}`);
}
