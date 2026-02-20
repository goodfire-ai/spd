# Example Neuron Labeling Prompt

This is an example prompt generated for neuron L31/N3732. Edit as needed.

---

```
You are analyzing a neuron in Llama-3.1-8B to understand its OUTPUT FUNCTION.

Neuron: L31/N3732 (Layer 31 of 31)
*** THIS IS THE FINAL LAYER - neurons here ONLY affect output logits directly ***
*** There are NO downstream neurons. 'routing' type is NOT applicable for L31. ***
Direct effect ratio: 100.0% (logit neuron)
Activates on (input pattern): word "b" in citation context of "b" highlighted text.
Suppressed by (input pattern): occurrences of "the" or words referring to specific aspects of the body or physical condition

=== OUTPUT EFFECTS (what this neuron DOES when it fires) ===

Static output projection (context-independent):
  PROMOTES:
    ' body'                   (+0.4029)
    ' Body'                   (+0.3789)
    'body'                    (+0.3528)
    'Body'                    (+0.3516)
    ' bodies'                 (+0.3260)
  SUPPRESSES:
    'anean'                   (-0.0349)
    'chten'                   (-0.0342)
    '.bz'                     (-0.0336)
    'くだ'                      (-0.0332)
    'iena'                    (-0.0325)

*** Downstream neurons with KNOWN functions ***
  ACTIVATES: "output token  body" (LOGIT/2547, w=+0.969)
  ACTIVATES: "output token  Body" (LOGIT/14285, w=+0.457)
  ACTIVATES: "output token  The" (LOGIT/578, w=+0.055)
  ACTIVATES: "output token  the" (LOGIT/279, w=+0.055)
  ACTIVATES: "output token  hunger" (LOGIT/34906, w=+0.043)

This is the FINAL LAYER (L31) - this neuron can ONLY affect output logits directly.
There are no downstream neurons. Focus entirely on the OUTPUT PROJECTION tokens above.
Look for semantic patterns in the tokens (medical terms, formatting, letter patterns, etc.).

(Note: The below guidance is for earlier layers, not L31)
Look at both the output projection AND downstream connections. The neuron may have a direct effect through the projection, indirect through the downstream connections, or both--look to see if there is a pattern across these.

Projection magnitude is relatively HIGH (0.40) - there's a decent chance this neuron has an interpretable pattern here.

Analyze the neuron using this procedure:
- Look at the tokens promoted and suppressed by the output projection. Assign this more weight when the projection magnitude is high, and also when the direct effect ratio is high.
- Look at the neurons activated and inhibited by this neuron. Is there a pattern? Does this related to the output projection?

Provide a structured response with these four fields:

1. INTERPRETABILITY: How confident are you that this neuron has a clear, UNIFYING pattern?
   - "high" = Clear, obvious unifying pattern (e.g., all promoted tokens are anatomy terms, or all start with "Ch-")
   - "medium" = Likely pattern but some noise mixed in (e.g., 4/5 tokens fit a theme, 1 is random)
   - "low" = No unifying pattern - tokens appear unrelated or random

   EXAMPLES:
   - HIGH: Suppresses [' muscle', ' abdominal', ' stomach', ' spinal'] → clear "anatomy terms" pattern
   - HIGH: Promotes [' Green', ' green', 'Green', 'GREEN'] → clear "green" pattern
   - MEDIUM: Promotes [' A', ' L', ' D', ' T', '.scalablytyped'] → mostly capital letters, one outlier
   - LOW: Promotes ['(水', '(日', 'jedn', '!\n\n\n\n', ' Vš'] → no unifying theme, just listing tokens

   IMPORTANT: If you cannot identify ONE unifying theme, mark as LOW.
   Do NOT describe neurons as "promotes X and Y and Z" listing unrelated tokens - that's LOW interpretability.

2. TYPE: What category best describes this neuron's OUTPUT effect on the vocabulary?
   - "semantic" = Promotes/suppresses tokens with shared meaning (medical terms, emotions, etc.)
   - "formatting" = Affects punctuation, whitespace, newlines, capitalization
   - "structural" = Sentence boundaries, list markers, section breaks
   - "lexical" = Letter/character patterns (e.g., words starting with "Ch-", or single letters)
   - "routing" = Works through downstream neurons (ONLY for layers < 31, NOT for final layer)
   - "associative" = Connects one concept (upstream) to another concept (downstream).
   - "unknown" = Cannot determine
   NOTE: TYPE describes the OUTPUT effect, not what triggers/activates the neuron.

3. SHORT_LABEL: A brief 3-8 word label summarizing the neuron's function.
   - If interpretable: describe the pattern (e.g., "capital-letter-promoter", "ellipsis-continuation-marker")
   - If low interpretability: write "uninterpretable" or "uninterpretable-routing"

4. OUTPUT_FUNCTION: A 1-2 sentence description of what this neuron DOES when it fires.
   - Focus on the OUTPUT effect (what tokens it promotes/suppresses)
   - A neuron's effect may be primarily PROMOTION, primarily SUPPRESSION, or both. Sometimes only one side has an interpretable pattern. If only one side is interpretable, focus on that.
   - Do NOT describe what activates the neuron (that's the input function, not output)
   - If uninterpretable, explain why (e.g., "No clear pattern in promoted tokens")

Format your response exactly like this:
INTERPRETABILITY: <high|medium|low>
TYPE: <semantic|formatting|structural|lexical|routing|unknown>
SHORT_LABEL: <your label>
OUTPUT_FUNCTION: <your description>
```

---

## Notes

- The prompt-building logic is in `scripts/interactive_labeling.py`, method `_build_output_label_prompt()` (lines 258-490)
- Context sections are built dynamically based on:
  - Layer (L31 gets special "final layer" guidance)
  - Direct effect ratio (logit vs routing neuron)
  - Whether downstream neurons have labels
  - Projection magnitude (high/moderate/low guidance)
- The response format is parsed by `parse_llm_response()` in the same file
