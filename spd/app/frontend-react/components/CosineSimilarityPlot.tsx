"use client";

interface CosineSimilarityPlotProps {
    title: string;
    data: number[][];
    indices: number[];
    disabledIndices: number[];
}

export default function CosineSimilarityPlot({ title, data, indices, disabledIndices }: CosineSimilarityPlotProps) {
    const isDisabled = (i: number, j: number): boolean => {
        return disabledIndices.includes(indices[i]) || disabledIndices.includes(indices[j]);
    };

    const getHeatmapColor = (value: number, i: number, j: number): string => {
        if (disabledIndices.includes(indices[i]) || disabledIndices.includes(indices[j])) {
            return "#f0f0f0";
        }

        const v = Math.max(0, Math.min(1, value));
        const whiteAmount = Math.round(255 * (1 - v));
        return `rgb(${whiteAmount}, ${whiteAmount}, 255)`;
    };

    return (
        <div className="my-4">
            <h4 className="m-0 mb-2 text-gray-800 text-sm">{title}</h4>
            <div className="flex gap-4 items-start">
                <div className="flex gap-1">
                    <div className="flex flex-col justify-around pr-1">
                        {indices.map((idx) => (
                            <div key={idx} className="text-xs text-gray-600 text-center min-w-[1.2em] h-5">
                                {idx}
                            </div>
                        ))}
                    </div>
                    <div className="flex flex-col">
                        {data.map((row, i) => (
                            <div key={i} className="flex">
                                {row.map((value, j) => (
                                    <div
                                        key={j}
                                        className={`w-5 h-5 border-[0.5px] border-black/10 cursor-pointer transition-transform ${
                                            isDisabled(i, j)
                                                ? "cursor-not-allowed hover:transform-none hover:border-black/10"
                                                : "hover:scale-110 hover:z-10 hover:border hover:border-gray-800"
                                        }`}
                                        style={{ backgroundColor: getHeatmapColor(value, i, j) }}
                                        title={`Components ${indices[i]} × ${indices[j]}: ${value.toFixed(3)}${isDisabled(i, j) ? " (disabled)" : ""}`}
                                    />
                                ))}
                            </div>
                        ))}
                    </div>
                </div>
            </div>
        </div>
    );
}