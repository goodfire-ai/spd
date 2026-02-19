# h.0.attn.o_proj:38

**Layer**: h.0.attn.o_proj  
**Firing density**: 18.37% (~1 in 5 tokens)  
**Mean CI**: 0.1542

**Old label**: [high] period and sentence starts  
**Old reasoning**: The component fires almost exclusively on the period token ('.' at 100% precision) or the token immediately following it. The output predictions focus on adverbs and discourse markers frequently used to begin sentences or transitions, such as 'suddenly', 'eventually', and 'unexpectedly'.

---

## Output function (what it predicts)

### Output precision
| Token | Precision |
|-------|-----------|
| [EOS] | 98% |
| wow | 96% |
| timidly | 95% |
| ironically | 94% |
| unexpectedly | 94% |
| incred | 91% |
| suddenly | 90% |
| eventually | 90% |
| later | 89% |
| curiously | 88% |
| ##aid | 87% |
| oh | 86% |

### Output PMI
| Token | PMI |
|-------|-----|
| [EOS] | 1.68 |
| wow | 1.65 |
| timidly | 1.65 |
| ironically | 1.63 |
| unexpectedly | 1.63 |
| incred | 1.60 |
| suddenly | 1.59 |
| eventually | 1.58 |
| later | 1.58 |
| curiously | 1.56 |
| ##aid | 1.55 |
| oh | 1.55 |

## Input function (what triggers it)

### Input recall
| Token | Recall |
|-------|--------|
| . | 41% |
| " | 9% |
| the | 6% |
| he | 3% |
| they | 3% |
| she | 3% |
| as | 2% |
| [EOS] | 2% |

### Input precision
| Token | Precision |
|-------|-----------|
| ##unched | 100% |
| . | 100% |
| inno | 100% |
| vis | 99% |
| disco | 98% |
| [EOS] | 97% |
| invis | 96% |
| ed | 96% |

### Input PMI
| Token | PMI |
|-------|-----|
| ##unched | 1.69 |
| . | 1.69 |
| inno | 1.69 |
| vis | 1.68 |
| disco | 1.67 |
| [EOS] | 1.66 |
| invis | 1.65 |
| ed | 1.65 |

## Activation examples (dual view)

**1a. fires on**: <<. it>> showed a path to a hidden treasure<<. his heart>> raced with excitement<<. he>> decided to follow the  
**1b. says**: . <<it showed>> a path to a hidden treasure. <<his heart raced>> with excitement. <<he decided>> to follow the

**2a. fires on**: courage to leave home, just like the birds<<. the>> garden has taught her that independence is a journey filled  
**2b. says**: courage to leave home, just like the birds. <<the garden>> has taught her that independence is a journey filled

**3a. fires on**: <<. she>> carried the lessons of the tree within her<<. she>> spoke to children about dreams and journeys<<. they>>  
**3b. says**: . <<she carried>> the lessons of the tree within her. <<she spoke>> to children about dreams and journeys. <<they>>

**4a. fires on**: <<she>> stood strong, but <<inside>> she felt small<<. every day>>, she listened to the breeze<<. ">> what  
**4b. says**: she <<stood>> strong, but inside <<she>> felt small. <<every day,>> she listened to the breeze. <<" what>>

**5a. fires on**: the lesson of trust back to the surface<<. [EOS] while>> the rain poured above, alice and her friend,  
**5b. says**: the lesson of trust back to the surface. <<[EOS] while the>> rain poured above, alice and her friend,

**6a. fires on**: he opened it and found a small, glowing creature<<. it>> had wings like a butterfly and eyes that sparkled  
**6b. says**: he opened it and found a small, glowing creature. <<it had>> wings like a butterfly and eyes that sparkled

**7a. fires on**: seen<<. it>> was perfect, and she was happy<<. friends>> surrounded her, laughter filled the air, and  
**7b. says**: seen. <<it was>> perfect, and she was happy. <<friends surrounded>> her, laughter filled the air, and

**8a. fires on**: he flew, he met a wise old owl<<. the>> owl told him that dreams can show you what you  
**8b. says**: he flew, he met a wise old owl. <<the owl>> told him that dreams can show you what you

**9a. fires on**: their special place, and now it felt empty<<. one>> night, <<while>> wandering, jose found a note under  
**9b. says**: their special place, and now it felt empty. <<one night>>, while <<wandering>>, jose found a note under

**10a. fires on**: leaves were gone, but his dreams were alive<<. he>> thought of how he would improve his machine next  
**10b. says**: leaves were gone, but his dreams were alive. <<he thought>> of how he would improve his machine next

**11a. fires on**: <<the>> wind was strong, and the sea was rough<<. they>> were fighting against the waves, their ship moving  
**11b. says**: the <<wind>> was strong, and the sea was rough. <<they were>> fighting against the waves, their ship moving

**12a. fires on**: walked in<<. jean>> dropped a cup<<. ">> what is this? " he thought<<. the>> alien  
**12b. says**: walked in. <<jean dropped>> a cup. <<" what>> is this? " he thought. <<the alien>>

**13a. fires on**: and knocked it over<<! the villagers>> gasped, but <<then>> everyone began to laugh<<. instead>> of sadness, jose  
**13b. says**: and knocked it over! <<the villagers gasped>>, but then <<everyone>> began to laugh. <<instead of>> sadness, jose

**14a. fires on**: trail in the sky<<. it>> looked like a comet<<! ">> this is my chance! " he shouted,  
**14b. says**: trail in the sky. <<it looked>> like a comet! <<" this>> is my chance! " he shouted,

**15a. fires on**: him<<. they>> laughed as they built the course together<<. it>> turned into a wild adventure with snow flying everywhere  
**15b. says**: him. <<they laughed>> as they built the course together. <<it turned>> into a wild adventure with snow flying everywhere

---

## Ideal interpretation

**Label**: says sentence-opening words after periods (subject pronouns, adverbs, [EOS])

**Reasoning process**:

1. **Start with output**: `[EOS]` (98%), `suddenly` (90%), `eventually` (90%), `later` (89%), `curiously` (88%). These are sentence-opening words: discourse adverbs, subject pronouns (via the `says` examples), and end-of-text markers. This component starts new sentences.

2. **Check input**: Input precision: `.` at 100%, `[EOS]` at 97%. Fires on sentence-ending punctuation. Recall: `.` at 41% — fires on 41% of all periods.

3. **Verify with examples**: The `says` lines show the sentence-start pattern beautifully: `. <<it showed>>`, `. <<his heart raced>>`, `. <<she carried>>`, `. <<every day>>`, `. <<it had>>`, `. <<friends surrounded>>`, `. <<the owl>>`. After every period, the component fires and produces the opening words of the next sentence.

4. **Synthesize**: This is a **sentence transition** component in the attention layer. It fires on periods and pushes the model to start a new sentence — with subject pronouns, temporal adverbs, or [EOS]. It's operating at a structural level, not semantic. Density is 18% because periods are common.

**This is a "structural" component**: Unlike the semantic (aquatic) and syntactic (pronouns) components, this one operates at the discourse level — managing sentence boundaries. The old label "period and sentence starts" was actually reasonable, but "says sentence-opening words after periods" is more precise about causality.

**Interesting observation**: For this kind of component, input and output are tightly coupled — the input IS the structural marker (period) and the output IS the structural action (start new sentence). There's no input/output confusion because the function is inherently structural. These are the easiest components to label correctly, and indeed the old label was close.

**Reasoning pattern for structural components**: When input = structural marker AND output = structural action AND they form a coherent narrative function (end sentence → start sentence), label the FUNCTION ("sentence transitions"), not just one side.
