# h.1.mlp.c_fc:564

**Layer**: h.1.mlp.c_fc  
**Firing density**: 0.95% (~1 in 105 tokens)  
**Mean CI**: 0.0066

**Old label**: [high] prepositions 'of', 'about', 'from'  
**Old reasoning**: The component fires almost exclusively on the prepositions 'of', 'about', 'from', and 'by'. It specifically predicts 'course' following 'of' (for 'of course') and other abstract nouns (inspiration, energy, hope) commonly used in prepositional phrases describing state or quality.

---

## Output function (what it predicts)

### Output precision
| Token | Precision |
|-------|-----------|
| course | 76% |
| afar | 46% |
| inspiration | 30% |
| paper | 24% |
| emot | 24% |
| energy | 23% |
| betrayal | 22% |
| history | 22% |
| hope | 22% |
| nature | 22% |
| faraway | 20% |
| loneliness | 19% |

### Output PMI
| Token | PMI |
|-------|-----|
| course | 4.38 |
| afar | 3.88 |
| inspiration | 3.47 |
| paper | 3.24 |
| emot | 3.23 |
| energy | 3.19 |
| betrayal | 3.16 |
| history | 3.16 |
| hope | 3.16 |
| nature | 3.13 |
| faraway | 3.04 |
| loneliness | 2.99 |

## Input function (what triggers it)

### Input recall
| Token | Recall |
|-------|--------|
| of | 71% |
| about | 14% |
| from | 8% |
| by | 4% |
| with | 0% |
| in | 0% |
| for | 0% |
| the | 0% |

### Input precision
| Token | Precision |
|-------|-----------|
| about | 64% |
| of | 55% |
| from | 37% |
| by | 30% |
| outsm | 4% |
| cal | 4% |
| fro | 4% |
| ##venge | 2% |

### Input PMI
| Token | PMI |
|-------|-----|
| about | 4.21 |
| of | 4.07 |
| from | 3.67 |
| by | 3.46 |
| outsm | 1.54 |
| cal | 1.47 |
| fro | 1.42 |
| ##venge | 0.94 |

## Activation examples (dual view)

**1a. fires on**: people saw each other. the boy had been betrayed <<by>> a close friend, so he wished for the orb  
**1b. says**: people saw each other. the boy had been betrayed by <<a>> close friend, so he wished for the orb

**2a. fires on**: tough, " she thought, feeling the heat <<of>> the sun. alice trekked through the sand  
**2b. says**: tough, " she thought, feeling the heat of <<the>> sun. alice trekked through the sand

**3a. fires on**: ##ing the magic of the earth. excited to learn <<from>> them, rita spent her days in the garden,  
**3b. says**: ##ing the magic of the earth. excited to learn from <<them>>, rita spent her days in the garden,

**4a. fires on**: hoverboard zoomed along, and they shared stories <<about>> their lives. alex learned that the woman was an  
**4b. says**: hoverboard zoomed along, and they shared stories about <<their>> lives. alex learned that the woman was an

**5a. fires on**: , except for a shy boy who felt out <<of>> place. he watched the colorful decorations and heard  
**5b. says**: , except for a shy boy who felt out of <<place>>. he watched the colorful decorations and heard

**6a. fires on**: to finding your place. " anne felt a spark <<of>> hope. she wanted to understand the stone ' s  
**6b. says**: to finding your place. " anne felt a spark of <<hope>>. she wanted to understand the stone ' s

**7a. fires on**: coins and old trinkets. he thought <<about>> taking some, but then he remembered the island '  
**7b. says**: coins and old trinkets. he thought about <<taking>> some, but then he remembered the island '

**8a. fires on**: top, he found an open window with a view <<of>> the sea. and there, sitting on the led  
**8b. says**: top, he found an open window with a view of <<the>> sea. and there, sitting on the led

**9a. fires on**: stroke brought her joy, and she felt the warmth <<of>> connection with the universe. she shared her work with  
**9b. says**: stroke brought her joy, and she felt the warmth of <<connection>> with the universe. she shared her work with

**10a. fires on**: ##ined a new friend. this holiday would be one <<of>> joy, not sadness. next year, he would  
**10b. says**: ##ined a new friend. this holiday would be one of <<joy>>, not sadness. next year, he would

**11a. fires on**: peace, alex took a moment to enjoy the beauty <<of>> the garden instead. he picked a single fruit,  
**11b. says**: peace, alex took a moment to enjoy the beauty of <<the>> garden instead. he picked a single fruit,

**12a. fires on**: by the river held a yearly festival. people <<from>> all around would come to dance, sing, and  
**12b. says**: by the river held a yearly festival. people from <<all>> around would come to dance, sing, and

**13a. fires on**: with it, a fearsome dragon. the people <<of>> the kingdom were scared. the king sent out a  
**13b. says**: with it, a fearsome dragon. the people of <<the>> kingdom were scared. the king sent out a

**14a. fires on**: kindness and tradition. kim felt grateful to be part <<of>> something so special. so, dear reader, remember  
**14b. says**: kindness and tradition. kim felt grateful to be part of <<something>> so special. so, dear reader, remember

**15a. fires on**: children, laughter mixing with the sun. the map <<of>> sorrow became a map <<of>> hope. leo learned that  
**15b. says**: children, laughter mixing with the sun. the map of <<sorrow>> became a map of <<hope>>. leo learned that

---

## Ideal interpretation

**Label**: says abstract/emotive nouns after prepositions ("of hope", "of course", "from afar")

**Reasoning process**:

1. **Start with output**: Top output tokens are abstract/emotive nouns: `course` (PMI=4.38!), `afar`, `inspiration`, `energy`, `betrayal`, `hope`, `nature`, `loneliness`. These are consistently abstract or emotionally-charged nouns.

2. **Check input**: Fires exclusively on prepositions: `of` (71% recall), `about` (14%), `from` (8%), `by` (4%). Very clean, tight input selectivity.

3. **Verify with examples**: The `says` lines are revealing but noisy. Some show the abstract noun pattern clearly: `of <<hope>>`, `of <<connection>>`, `of <<joy>>`, `of <<sorrow>>`, `of <<something>>`, `of <<place>>`. Others show common determiners: `of <<the>>`, `by <<a>>`, `from <<them>>`. The abstract nouns appear alongside common words because prepositions are frequently followed by both.

4. **Synthesize function**: This component fires on prepositions and biases the model toward predicting abstract/emotive completions over concrete ones. When you see "a spark of ___", this component pushes toward "hope" rather than "light". It's a **semantic steering** component — it doesn't just predict the syntactic category (noun after preposition), it biases WHICH nouns toward the abstract/emotive.

5. **Note the interesting case**: `of <<course>>` has by far the highest output PMI (4.38) — this is a fixed phrase ("of course") where the preposition deterministically predicts the completion. The component may be handling both fixed prepositional phrases AND abstract noun selection.

**What the old label got wrong**: "prepositions 'of', 'about', 'from'" perfectly describes the input but says nothing about the output. The old reasoning actually nails it — it mentions "course" and "abstract nouns" — but the label drops all of this.

**This is harder than case 1**: The output pattern is noisier (some examples just show `the`, `a`). A good interpreter needs to look past the noise and notice that the HIGH-PMI outputs are the abstract nouns, not the common words. The output PMI table is essential here — without it, you'd think the output is just "common words after prepositions."
