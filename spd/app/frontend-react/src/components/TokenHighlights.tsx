import React, { useMemo } from 'react';

interface Segment {
    text: string;
    ciValue: number;
    isActive: boolean;
}

interface TokenHighlightsProps {
    rawText: string;
    offsetMapping: [number, number][];
    tokenCiValues: number[];
    activePosition?: number;
    precision?: number;
    getHighlightColor?: (importance: number) => string;
}

const defaultHighlightColor = (importance: number): string => {
    const clamped = Math.min(Math.max(importance, 0), 1);
    const opacity = 0.15 + clamped * 0.35;
    return `rgba(0, 200, 0, ${opacity})`;
};

export const TokenHighlights: React.FC<TokenHighlightsProps> = ({
    rawText,
    offsetMapping,
    tokenCiValues,
    activePosition = -1,
    precision = 3,
    getHighlightColor = defaultHighlightColor,
}) => {
    const segments = useMemo(() => {
        const result: Segment[] = [];
        let cursor = 0;

        offsetMapping.forEach(([start, end], idx) => {
            if (cursor < start) {
                result.push({ text: rawText.slice(cursor, start), ciValue: 0, isActive: false });
            }

            const tokenText = rawText.slice(start, end);
            const ciValue = tokenCiValues[idx] ?? 0;
            result.push({ text: tokenText, ciValue, isActive: idx === activePosition });

            cursor = end;
        });

        if (cursor < rawText.length) {
            result.push({ text: rawText.slice(cursor), ciValue: 0, isActive: false });
        }

        return result.filter((segment) => segment.text.length > 0);
    }, [rawText, offsetMapping, tokenCiValues, activePosition]);

    return (
        <span className="token-highlights">
            {segments.map((segment, idx) => {
                if (segment.ciValue > 0) {
                    return (
                        <span
                            key={idx}
                            className={`token-highlight ${segment.isActive ? 'active-token' : ''}`}
                            style={{ backgroundColor: getHighlightColor(segment.ciValue) }}
                            title={`Importance: ${segment.ciValue.toFixed(precision)}`}
                        >
                            {segment.text}
                        </span>
                    );
                }
                return <span key={idx}>{segment.text}</span>;
            })}
        </span>
    );
};
