# h.1.attn.k_proj:143

**Layer**: h.1.attn.k_proj  
**Firing density**: 3.76% (~1 in 26 tokens)  
**Mean CI**: 0.0271

**Old label**: [high] moral conflict and betrayal  
**Old reasoning**: The component fires with overwhelmingly high precision on specific tokens associated with negative moral actions or emotional consequences such as 'scratched', 'guilt', 'betrayed', and 'stole'. The output predictions ('gripped', 'creeping') further support a context of escalating tension or emotional fallout from these actions.

---

## Output function (what it predicts)

### Output precision
| Token | Precision |
|-------|-----------|
| gripped | 62% |
| ##elling | 52% |
| flooded | 49% |
| ##mates | 49% |
| keys | 46% |
| creeping | 45% |
| wash | 44% |
| ##oned | 42% |
| hunt | 41% |
| creep | 40% |
| ##ild | 40% |
| ##sw | 39% |

### Output PMI
| Token | PMI |
|-------|-----|
| gripped | 2.81 |
| ##elling | 2.63 |
| flooded | 2.57 |
| ##mates | 2.57 |
| keys | 2.50 |
| creeping | 2.48 |
| wash | 2.45 |
| ##oned | 2.40 |
| hunt | 2.40 |
| creep | 2.37 |
| ##ild | 2.36 |
| ##sw | 2.33 |

## Input function (what triggers it)

### Input recall
| Token | Recall |
|-------|--------|
| help | 3% |
| treasure | 2% |
| lost | 2% |
| love | 2% |
| joy | 1% |
| dreams | 1% |
| laughter | 1% |
| find | 1% |

### Input precision
| Token | Precision |
|-------|-----------|
| scratched | 97% |
| plotted | 97% |
| guilt | 96% |
| betrayed | 95% |
| tricked | 94% |
| stole | 93% |
| uneasy | 92% |
| argued | 91% |

### Input PMI
| Token | PMI |
|-------|-----|
| scratched | 3.25 |
| plotted | 3.25 |
| guilt | 3.23 |
| betrayed | 3.23 |
| tricked | 3.22 |
| stole | 3.21 |
| uneasy | 3.20 |
| argued | 3.19 |

## Activation examples (dual view)

**1a. fires on**: " the boy said, " can you <<help>> me <<find>> my <<lost puppy>>? " the dragon snorted  
**1b. says**: " the boy said, " can you help <<me>> find <<my>> lost <<puppy?>> " the dragon snorted

**2a. fires on**: <<her>> waiting for him. " will <<she>> understand my <<choices>>? " he wondered. " or will <<she>> think  
**2b. says**: her <<waiting>> for him. " will she <<understand>> my choices<<?>> " he wondered. " or will she <<think>>

**3a. fires on**: . one day, a girl named kim decided to <<explore>> it. she loved acting and dreamed of bringing  
**3b. says**: . one day, a girl named kim decided to explore <<it>>. she loved acting and dreamed of bringing

**4a. fires on**: instead of <<candy>>, she decided to create her own <<adventure>>. she rode around town, making up stories about  
**4b. says**: instead of candy<<,>> she decided to create her own adventure<<.>> she rode around town, making up stories about

**5a. fires on**: comb, laughing as he approached her. <<relief>> flooded <<maria>>. she felt the worry leave her. " you  
**5b. says**: comb, laughing as he approached her. relief <<flooded>> maria<<.>> she felt the worry leave her. " you

**6a. fires on**: each discovery, their <<love>> grew stronger. the lost <<city>> became a <<canvas>> for their <<dreams>>, filled with <<laughter>>  
**6b. says**: each discovery, their love <<grew>> stronger. the lost city <<became>> a canvas <<for>> their dreams<<,>> filled with laughter

**7a. fires on**: " with that, we set off to find a <<magic>> gem that could save her home. together,  
**7b. says**: " with that, we set off to find a magic <<gem>> that could save her home. together,

**8a. fires on**: stone <<transform>>s into a tiny star. " i <<lost>> my light when the sky flipped. but your <<courage>>  
**8b. says**: stone transform<<s>> into a tiny star. " i lost <<my>> light when the sky flipped. but your courage

**9a. fires on**: , " it read, a symbol of <<love>> and <<trust>>. but that <<trust>> was shattered, leaving only spl  
**9b. says**: , " it read, a symbol of love <<and>> trust<<.>> but that trust <<was>> shattered, leaving only spl

**10a. fires on**: they both shared a knowing look, an unsp<<oken understanding>> of <<loss>> hanging in the air. they  
**10b. says**: they both shared a knowing look, an unspoken <<understanding of>> loss <<hang>>ing in the air. they

**11a. fires on**: brighter. lanterns lit up the square, and the <<music>> grew louder. the girl danced, her feet moving  
**11b. says**: brighter. lanterns lit up the square, and the music <<grew>> louder. the girl danced, her feet moving

**12a. fires on**: all she felt was <<guilt>>. the other dinosaurs were <<angry>>. " what if they find me? " she  
**12b. says**: all she felt was guilt<<.>> the other dinosaurs were angry<<.>> " what if they find me? " she

**13a. fires on**: whole week <<arg>>uing about who was the better <<artist>>. one day, the younger sister <<painted>> a big  
**13b. says**: whole week arg<<u>>ing about who was the better artist<<.>> one day, the younger sister painted <<a>> big

**14a. fires on**: but pressed on. she had heard tales of the <<beast>>. but she was ready. she had trained  
**14b. says**: but pressed on. she had heard tales of the beast<<.>> but she was ready. she had trained

**15a. fires on**: ##lociraptor, who was quick and <<smart>>, agreed. they ran toward the light, excited  
**15b. says**: ##lociraptor, who was quick and smart<<,>> agreed. they ran toward the light, excited

---

## Ideal interpretation

**Label**: unclear — diffuse pattern, possibly says tension/intensity verbs

**Reasoning process**:

1. **Start with output**: `gripped` (62%), `flooded` (49%), `creeping` (45%), `creep` (40%), `hunt` (41%). There's a loose theme of physical intensity/tension verbs, but precision is only 40-62% and the subword fragments (`##elling`, `##mates`, `##oned`, `##ild`, `##sw`) are hard to interpret.

2. **Check input**: Input precision is dramatic — `scratched` 97%, `plotted` 97%, `guilt` 96%, `betrayed` 95%, `tricked` 94%, `stole` 93%. Very tight input selectivity on moral conflict / negative action words. But recall is low: `help` 3%, `treasure` 2%, `lost` 2%. The high-recall tokens are positive words!

3. **The contradiction**: Input PMI says "fires on moral conflict words" but input recall says the most COMMON firing tokens are positive words like `help`, `love`, `joy`, `dreams`, `laughter`. This is a k_proj component with 3.7% density — it fires on many tokens but has especially high precision on conflict words.

4. **Verify with examples**: The `says` lines don't show a clear pattern. Outputs include: `<<me>>`, `<<waiting>>`, `<<it>>`, punctuation, `<<flooded>>`, `<<grew>>`, `<<gem>>`. Very diffuse. No consistent theme.

5. **Synthesize**: This is genuinely ambiguous. The input has a clear theme (moral conflict words have very high precision) but the output is diffuse. The component may be doing something at the attention level (k_proj = key projection) that's harder to capture as "predicts X" — it might be affecting WHAT other positions attend to rather than directly predicting tokens.

**This is the right call for "unclear"**: The output-centric strategy correctly returned "unclear" for this component. The old label "moral conflict and betrayal" describes the input selectivity but overstates the coherence — the output doesn't support a tight interpretation.

**Meta-observation**: Attention components (especially k_proj and q_proj) may be harder to interpret via token prediction because they operate on the attention pattern, not directly on token probabilities. Their "output function" is mediated through what tokens get attended to, which is several steps removed from next-token prediction. A different interpretation frame may be needed for these — something about "what information gets routed where" rather than "what token gets predicted."
