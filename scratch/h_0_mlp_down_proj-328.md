# h.0.mlp.down_proj:328

**Layer**: h.0.mlp.down_proj  
**Firing density**: 4.45% (~1 in 22 tokens)  
**Mean CI**: 0.0347

**Old label**: [high] mid-word subword completions  
**Old reasoning**: The component consistently triggers on the initial stems or mid-sections of complex words (e.g., 'exper', 'perfor', 'complet') and specifically predicts Latinate/morphological suffixes like '##ations', '##ibility', '##ivated', and '##ings'. The activation examples show it firing on word fragments that require specific structural completion.

---

## Output function (what it predicts)

### Output precision
| Token | Precision |
|-------|-----------|
| ##essed | 96% |
| ##ind | 94% |
| ##ivated | 94% |
| ##ings | 93% |
| ##ations | 93% |
| ##iced | 93% |
| ##rance | 92% |
| ##vous | 92% |
| ##ibility | 92% |
| ##eter | 92% |
| ##ention | 92% |
| ##ded | 92% |

### Output PMI
| Token | PMI |
|-------|-----|
| ##essed | 3.07 |
| ##ind | 3.05 |
| ##ivated | 3.05 |
| ##ings | 3.04 |
| ##ations | 3.04 |
| ##iced | 3.04 |
| ##rance | 3.03 |
| ##vous | 3.03 |
| ##ibility | 3.03 |
| ##eter | 3.03 |
| ##ention | 3.03 |
| ##ded | 3.03 |

## Input function (what triggers it)

### Input recall
| Token | Recall |
|-------|--------|
| moment | 2% |
| full | 2% |
| out | 1% |
| proud | 1% |
| instead | 1% |
| part | 1% |
| ##s | 1% |
| ##t | 1% |

### Input precision
| Token | Precision |
|-------|-----------|
| fri | 100% |
| trea | 100% |
| spec | 100% |
| / | 100% |
| fr | 100% |
| pow | 100% |
| ##ather | 100% |
| beh | 100% |

### Input PMI
| Token | PMI |
|-------|-----|
| inspir | 3.11 |
| pict | 3.11 |
| dan | 3.11 |
| len | 3.11 |
| decor | 3.11 |
| fri | 3.11 |
| piec | 3.11 |
| fr | 3.11 |

## Activation examples (dual view)

**1a. fires on**: remembered his grandfather ' s words about trusting inst<<inct>>s. <<instead>> of running, leo slowly backed  
**1b. says**: remembered his grandfather ' s words about trusting instin<<cts>>. instead <<of>> running, leo slowly backed

**2a. fires on**: . when she looked into it, she saw the <<faces>> of those who had su<<ffer>>ed <<because>> of her  
**2b. says**: . when she looked into it, she saw the faces <<of>> those who had suffer<<ed>> because <<of>> her

**3a. fires on**: " the boy jumped <<out>> of <<bed>>, <<wide>> - <<eye>>d. " what was that? " he asked  
**3b. says**: " the boy jumped out <<of>> bed<<,>> wide <<->> eye<<d>>. " what was that? " he asked

**4a. fires on**: the friends felt sad. then, anne remembered the <<kind>> voice of lily. she sang a <<song>> that filled  
**4b. says**: the friends felt sad. then, anne remembered the kind <<voice>> of lily. she sang a song <<that>> filled

**5a. fires on**: stars shine differently. suddenly, a shooting star <<bl>>azed across the sky, <<illumin>>ating the darkness  
**5b. says**: stars shine differently. suddenly, a shooting star bl<<a>>zed across the sky, illumin<<ating>> the darkness

**6a. fires on**: holding her breath, she thought about everything she had <<exper>>ienced. she  
**6b. says**: holding her breath, she thought about everything she had exper<<ience>>d. she

**7a. fires on**: silly jokes and made plans. the girl felt a <<spark>> of hope. with each laugh, she felt lighter  
**7b. says**: silly jokes and made plans. the girl felt a spark <<of>> hope. with each laugh, she felt lighter

**8a. fires on**: he pushed on, feeling a <<mix>> of fear and <<thrill>>. the <<stairs>> opened into a vast cave filled with  
**8b. says**: he pushed on, feeling a mix <<of>> fear and thrill<<.>> the stairs <<opened>> into a vast cave filled with

**9a. fires on**: glowing <<pools>> of water. each <<pool>> reflected strange <<images>> of people and places. taking a <<moment>> to catch  
**9b. says**: glowing pool<<s of>> water. each pool <<reflected>> strange images <<of>> people and places. taking a moment <<to>> catch

**10a. fires on**: ice stone, but his heart was <<full>>. the <<surprise>> of the glowing river taught him that life is about  
**10b. says**: ice stone, but his heart was full<<.>> the surprise <<of>> the glowing river taught him that life is about

**11a. fires on**: joined in. the sky turned into a giant <<dance floor>>, filled with laughter. everyone had so much fun  
**11b. says**: joined in. the sky turned into a giant dance <<floor,>> filled with laughter. everyone had so much fun

**12a. fires on**: his art. the villagers thought him a geni<<us>>, <<bl>>ind to his evil. he built a  
**12b. says**: his art. the villagers thought him a genius<<,>> bl<<ind>> to his evil. he built a

**13a. fires on**: <<complet>>ed a silly invention : a dancing robot <<made>> of scr<<aps>>. the jester turned it on,  
**13b. says**: complet<<ed>> a silly invention : a dancing robot made <<of>> scraps<<.>> the jester turned it on,

**14a. fires on**: something unique. she realized she had to overcome the <<fear>> of others ' opinions. with each brush  
**14b. says**: something unique. she realized she had to overcome the fear <<of>> others ' opinions. with each brush

**15a. fires on**: with alex. they learned that every turtle, no <<matter>> the age, has something special to give.  
**15b. says**: with alex. they learned that every turtle, no matter <<the>> age, has something special to give.

---

## Ideal interpretation

**What I see**: This component is doing morphological completion. It fires on word stems and predicts their suffixes.

Input: word stems and fragments — `inspir[ation]`, `pict[ure]`, `dan[ce]`, `decor[ation]`, `fri[end]`, `piec[e]`, `spec[ial]`, `pow[er]`, `beh[ind]`. Also fires on whole words that happen to end mid-morpheme boundary: `moment`, `full`, `out`, `instead`.

Output: suffixes — `##essed`, `##ivated`, `##ings`, `##ations`, `##rance`, `##ibility`, `##ention`, `##ded`. These are all Latinate/complex English suffixes.

Examples confirm it: `inst<<inct>>` → `instin<<cts>>`, `su<<ffer>>` → `suffer<<ed>>`, `<<exper>>` → `exper<<ience>>`, `<<bl>>azed` → `bl<<a>>zed`, `<<illumin>>` → `illumin<<ating>>`, `geni<<us>>` → `genius<<,>>`.

**This is purely mechanical**: there's no semantic content here, no narrative-level state, no interesting input/output divergence. The component completes multi-token words. It operates at the sub-word level — below semantics, below syntax, just morphology.

**Label**: completes multi-token words (stems → suffixes)

**The old label was fine**: "mid-word subword completions" captures this well. The old reasoning was also good. This is one of those components where the function is straightforward and a simple description works.

**Prompt design implication**: The interpreter should be allowed to give simple, mechanical labels when the pattern IS simple. Not everything needs to be described in terms of semantic content or narrative function. Some components are just doing string completion.
