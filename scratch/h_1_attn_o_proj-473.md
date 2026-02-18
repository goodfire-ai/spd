# h.1.attn.o_proj:473

**Layer**: h.1.attn.o_proj  
**Firing density**: 1.51% (~1 in 66 tokens)  
**Mean CI**: 0.0089

**Old label**: [high] water and aquatic contexts  
**Old reasoning**: The component fires on terms describing liquid movement (splashing, flowing, stirred) and strongly predicts water-related nouns like frog, fish, pond, and riverbank.

---

## Output function (what it predicts)

### Output precision
| Token | Precision |
|-------|-----------|
| ##bbit | 74% |
| frog | 71% |
| water | 71% |
| fish | 64% |
| surface | 64% |
| ##ong | 58% |
| riverbank | 56% |
| river | 55% |
| pond | 55% |
| frogs | 54% |
| lake | 54% |
| ripp | 54% |

### Output PMI
| Token | PMI |
|-------|-----|
| ##bbit | 3.89 |
| frog | 3.85 |
| water | 3.85 |
| fish | 3.75 |
| surface | 3.75 |
| ##ong | 3.65 |
| riverbank | 3.61 |
| river | 3.59 |
| pond | 3.58 |
| frogs | 3.58 |
| lake | 3.58 |
| ripp | 3.57 |

## Input function (what triggers it)

### Input recall
| Token | Recall |
|-------|--------|
| the | 44% |
| a | 6% |
| and | 4% |
| she | 3% |
| he | 3% |
| they | 2% |
| , | 2% |
| to | 2% |

### Input precision
| Token | Precision |
|-------|-----------|
| splashing | 91% |
| splashed | 75% |
| ##ety | 44% |
| fishing | 42% |
| ##llow | 39% |
| flowing | 31% |
| cool | 27% |
| rough | 26% |

### Input PMI
| Token | PMI |
|-------|-----|
| splashing | 4.09 |
| splashed | 3.91 |
| ##ety | 3.36 |
| fishing | 3.34 |
| ##llow | 3.25 |
| flowing | 3.03 |
| cool | 2.89 |
| rough | 2.85 |

## Activation examples (dual view)

**1a. fires on**: himself at <<the>> start of his life. he watched <<the>> love of his parents and felt their joy. in  
**1b. says**: himself at the <<start>> of his life. he watched the <<love>> of his parents and felt their joy. in

**2a. fires on**: " this feels magical! " she whispered. as <<the>> boat sailed further, <<the>> water started <<to>> swirl around  
**2b. says**: " this feels magical! " she whispered. as the <<boat>> sailed further, the <<water>> started to <<swirl>> around

**3a. fires on**: he make her heart flutter? one day, as <<the>> leaves fell around them, alex decided to show jean  
**3b. says**: he make her heart flutter? one day, as the <<leaves>> fell around them, alex decided to show jean

**4a. fires on**: . she felt joy and warmth, knowing she found <<the best>> ice cream. [EOS] fleeting time pushed  
**4b. says**: . she felt joy and warmth, knowing she found the <<best ice>> cream. [EOS] fleeting time pushed

**5a. fires on**: " i ' m standing! " he shouted. <<the>> world around him blurred, and he rode <<the>>  
**5b. says**: " i ' m standing! " he shouted. the <<world>> around him blurred, and he rode the

**6a. fires on**: loss. days turned into weeks, and mia visited <<the>> enchanted forest every day. she hoped to find leo  
**6b. says**: loss. days turned into weeks, and mia visited the <<enchanted>> forest every day. she hoped to find leo

**7a. fires on**: . one day, a golden stone was discovered. <<this>> stone was said to have the power to control the  
**7b. says**: . one day, a golden stone was discovered. this <<stone>> was said to have the power to control the

**8a. fires on**: find something special! " she said. they reached <<the>> door, and <<the>> girl inserted the key  
**8b. says**: find something special! " she said. they reached the <<door>>, and the <<girl>> inserted the key

**9a. fires on**: full of hope. one day, the river flooded<<. the water rushed through the>> streets<<.>> people were scared  
**9b. says**: full of hope. one day, the river flooded. <<the water rushed through the streets>>. <<people>> were scared

**10a. fires on**: , <<a>> dark figure emerged. <<the>> farmer and <<the>> nymph froze, eyes wide in horro  
**10b. says**: , a <<dark>> figure emerged. the <<far>>mer and the <<n>>ymph froze, eyes wide in horro

**11a. fires on**: they felt a bond form, a wonderful connection across <<time>>. after a long talk, the ghost smiled.  
**11b. says**: they felt a bond form, a wonderful connection across time<<.>> after a long talk, the ghost smiled.

**12a. fires on**: , <<a>> grace<<ful>> mermaid emerged, her <<hair>> flowing <<like>> seaweed. the mermaid had bright scales that shimmered in  
**12b. says**: , a <<grace>>ful <<mermaid>> emerged, her hair <<flowing>> like <<seaweed>>. the mermaid had bright scales that shimmered in

**13a. fires on**: <<he>> felt a rush of joy. <<the>> fish was <<a>> symbol of his patience and courage. <<he>> released  
**13b. says**: he <<felt>> a rush of joy. the <<fish>> was a <<symbol>> of his patience and courage. he <<relea>>sed

**14a. fires on**: , thinking he was alone. just as he reached <<the>> pond, <<the>> keepers jumped out and surrounded him  
**14b. says**: , thinking he was alone. just as he reached the <<pond>>, the <<keeper>>s jumped out and surrounded him

**15a. fires on**: <<the>> stone began to change. <<it>> turned into a <<small>> dragon! the dragon was small and looked weak,  
**15b. says**: the <<stone>> began to change. it <<turned>> into a small <<dragon>>! the dragon was small and looked weak,

---

## Ideal interpretation

**Label**: says water/nature nouns (frog, fish, river, pond) in aquatic contexts

**Reasoning process**:

1. **Start with output**: Very clean semantic cluster: `frog`, `water`, `fish`, `riverbank`, `river`, `pond`, `lake`, `frogs`, `ripp[le]`. All aquatic/nature nouns. Also `##bbit` (rabbit) and `surface` — nature-adjacent. Output precision is 55-74% for these — high for semantic features.

2. **Check input**: Input is interesting. Recall is dominated by `the` (44%) — this component fires after `the` a lot. But input PRECISION tells the real story: `splashing` (91%), `splashed` (75%), `fishing` (42%), `flowing` (31%). It fires in aquatic CONTEXTS, on many tokens, but particularly on water-related verbs/adjectives.

3. **Verify with examples**: The `says` lines show nature nouns: `<<boat>>`, `<<water>>`, `<<swirl>>`, `<<leaves>>`, `<<enchanted>>`, `<<stone>>`, `<<pond>>`, `<<fish>>`, `<<dragon>>`, `<<mermaid>>`, `<<seaweed>>`. The aquatic theme is clear but bleeds into broader nature/fantasy.

4. **Note the interesting pattern**: This is an ATTENTION component (o_proj), not MLP. It fires mostly on `the` — a determiner. Its job is: when `the` appears in a context that's been discussing water/nature, predict water/nature nouns as the next word. The attention mechanism is reading the context to determine WHICH kind of noun to predict after `the`.

5. **Synthesize**: This is a semantic steering component, but specifically one that operates via attention. It reads aquatic context from earlier in the sequence and biases the prediction of the NEXT noun toward water/nature words. Label should capture both the semantic domain and the mechanism.

**The old label was actually decent here**: "water and aquatic contexts" captures the right semantic domain. But it doesn't distinguish input from output — is the component responding to water contexts (input) or producing water tokens (output)? In fact it's both, but the causal direction matters for editing: ablating this suppresses water-word PRODUCTION.

**This component demonstrates why semantic labels need the input→output structure**: The same component can have input selectivity (aquatic context) and output effect (aquatic nouns) in the same semantic domain. The function is "when in a water scene, reinforce water vocabulary" — not just "water."
