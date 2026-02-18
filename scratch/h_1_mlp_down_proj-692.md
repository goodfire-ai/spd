# h.1.mlp.down_proj:692

**Layer**: h.1.mlp.down_proj  
**Firing density**: 21.55% (~1 in 4 tokens)  
**Mean CI**: 0.1774

**Old label**: [high] third-person object pronouns  
**Old reasoning**: The component shows nearly 100% output precision for tokens 'them', 'him', 'us', and 'me'. The activation examples consistently fire on verbs followed by object pronouns (e.g., 'told him', 'saw him', 'wished for them') or pronouns occurring in prepositional phrases.

---

## Output function (what it predicts)

### Output precision
| Token | Precision |
|-------|-----------|
| them | 99% |
| him | 99% |
| us | 98% |
| afar | 96% |
| me | 95% |
| whatever | 95% |
| circles | 94% |
| himself | 94% |
| diamonds | 94% |
| herself | 93% |
| oneself | 93% |
| themselves | 92% |

### Output PMI
| Token | PMI |
|-------|-----|
| them | 1.53 |
| him | 1.53 |
| us | 1.51 |
| afar | 1.50 |
| me | 1.49 |
| whatever | 1.48 |
| circles | 1.48 |
| himself | 1.47 |
| diamonds | 1.47 |
| herself | 1.46 |
| oneself | 1.46 |
| themselves | 1.46 |

## Input function (what triggers it)

### Input recall
| Token | Recall |
|-------|--------|
| of | 6% |
| in | 5% |
| with | 4% |
| to | 3% |
| and | 3% |
| for | 2% |
| on | 1% |
| at | 1% |

### Input precision
| Token | Precision |
|-------|-----------|
| 4 | 100% |
| gent | 100% |
| shimm | 100% |
| / | 100% |
| + | 100% |
| f | 100% |
| ( | 100% |
| among | 100% |

### Input PMI
| Token | PMI |
|-------|-----|
| ##ough | 1.53 |
| butterf | 1.53 |
| thanked | 1.53 |
| / | 1.53 |
| $ | 1.53 |
| beneath | 1.53 |
| ( | 1.53 |
| grabbed | 1.53 |

## Activation examples (dual view)

**1a. fires on**: but she couldn ' t <<forget>> her task. she <<told him>> she had to <<leave>>, but he <<grabbed>> her  
**1b. says**: but she couldn ' t forget <<her>> task. she told <<him she>> had to leave<<,>> but he grabbed <<her>>

**2a. fires on**: . they <<chirped along with>> her and <<filled>> her heart <<with>> joy. " the gnome smiled. " <<sometimes>>  
**2b. says**: . they chirped <<along with her>> and filled <<her>> heart with <<joy>>. " the gnome smiled. " sometimes

**3a. fires on**: laughter <<with>> friends, games <<in>> the sun, <<and quiet>> moments <<with>> his family. as the sun set,  
**3b. says**: laughter with <<friends>>, games in <<the>> sun, and <<quiet moments>> with <<his>> family. as the sun set,

**4a. fires on**: coral. excitedly, he swam <<to>> it and gra<<sped>> the key tightly <<in>> his hand. this <<was>>  
**4b. says**: coral. excitedly, he swam to <<it>> and grasp<<ed the>> key tightly in <<his>> hand. this was

**5a. fires on**: the waves. the irony <<of>> her wish <<hung>> heavy <<in>> the air <<->> her desire to <<help only brought>> pain  
**5b. says**: the waves. the irony of <<her>> wish hung <<heavy>> in <<the>> air - <<her>> desire to help <<only brought pain>>

**6a. fires on**: not def<<ine worth>>. he still <<had>> his dreams <<and>> his voice. one rainy evening, he <<found>> an  
**6b. says**: not define <<worth.>> he still had <<his>> dreams and <<his>> voice. one rainy evening, he found <<an>>

**7a. fires on**: is not just <<about>> food ; it is <<about>> hearts <<connecting>>. [EOS] <<behind>> the old barn, a girl  
**7b. says**: is not just about <<food>> ; it is about <<hearts>> connecting<<.>> [EOS] behind <<the>> old barn, a girl

**8a. fires on**: team <<spread out across>> the city, disgu<<ising themselves in>> cas<<ual>> clothes. they looked <<for>>  
**8b. says**: team spread <<out across the>> city, disguising <<themselves in ca>>sual <<clothes>>. they looked for

**9a. fires on**: small, <<trapped>> animal. it was a rabbit caught <<in>> a net. <<without thinking>>, the boy rushed <<over>>  
**9b. says**: small, trapped <<animal>>. it was a rabbit caught in <<a>> net. without <<thinking,>> the boy rushed over

**10a. fires on**: <<and>> fear. but the shadow only <<grew>> darker, <<revealing>> a painful truth. the spell was a trick  
**10b. says**: and <<fear>>. but the shadow only grew <<darker>>, revealing <<a>> painful truth. the spell was a trick

**11a. fires on**: <<wished for>> them to <<look to>> the stars and <<know that in>> every dark moment, they could <<find>> light <<within>>  
**11b. says**: wished <<for them>> to look <<to the>> stars and know <<that in every>> dark moment, they could find <<light>> within

**12a. fires on**: , " they said. samuel and lena <<decided>> to <<play>> games and <<make>> everyone laugh. slowly, smiles returned  
**12b. says**: , " they said. samuel and lena decided <<to>> play <<games>> and make <<everyone>> laugh. slowly, smiles returned

**13a. fires on**: , but they also talked <<about what>> it meant <<to take>> risks. their friendship blossomed <<into>> something deeper  
**13b. says**: , but they also talked about <<what it>> meant to <<take ri>>sks. their friendship blossomed into <<something>> deeper

**14a. fires on**: eyes sparkled <<when>> she <<saw>> him. " <<are>> you <<searching for>> the lost city? " she <<asked>>, her  
**14b. says**: eyes sparkled when <<she>> saw <<him>>. " are <<you>> searching <<for the>> lost city? " she asked<<,>> her

**15a. fires on**: <<determined>> to <<bring back>> the light, the child ran <<to>> the tallest tree. they <<called on>> the creatures <<around>>  
**15b. says**: determined <<to>> bring <<back the>> light, the child ran to <<the>> tallest tree. they called <<on the>> creatures around

---

## Ideal interpretation

**Label**: says object pronouns and reflexives (them, him, her, himself, herself)

**Reasoning process**:

1. **Start with output**: Output precision is extraordinary — 99% for `them`, 99% for `him`, 98% for `us`, 95% for `me`, 94% for `himself`, 93% for `herself`. This is about as clean as it gets. The component's output function is overwhelmingly "produce object/reflexive pronouns."

2. **Check input**: Input is very broad — fires 21.5% of the time (1 in 4 tokens). Input recall is spread across common function words: `of` 6%, `in` 5%, `with` 4%, `to` 3%. Input precision shows 100% for rare tokens but these are noise from the high density. The input function is essentially "fires very broadly, almost everywhere."

3. **Verify with examples**: The `says` lines do show pronouns: `<<her>>`, `<<him>>`, `<<his>>`, etc. But because this component fires so densely (21.5%), the examples are noisy — it fires on almost everything and there are many non-pronoun tokens in the output positions too.

4. **Synthesize function**: This is a **high-density baseline component** that broadly fires and pushes the model toward producing object pronouns. It doesn't have a tight input selectivity — its job is more like a persistent bias. The output precision being 99% despite 21.5% density is remarkable: this component is responsible for nearly ALL object pronoun predictions in the model.

**This is an interesting component type**: Unlike the first two cases, input and output function are essentially decoupled. The input is "fire almost everywhere" and the output is "bias toward pronouns." The old label "third-person object pronouns" was actually quite good because for this type of component, the output IS the function — there's no interesting input pattern to contrast with.

**Diagnostic note**: The `says` view is LESS helpful here than for sparse components. With 21.5% density, the shifted highlights are everywhere and hard to interpret visually. The output precision table is the decisive evidence. This is a case where the interpreter should weight the statistical data more heavily than the examples.
