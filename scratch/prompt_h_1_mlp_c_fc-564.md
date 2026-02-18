Describe what this neural network component does.

Each component has an input function (what causes it to fire) and an output function (what tokens it causes the model to produce). These are often different — a component might fire on periods but produce sentence-opening words, or fire on prepositions but produce abstract nouns.

## Context
- Model: LlamaSimpleMLP (2 blocks), dataset: SimpleStories: 2M+ short stories (200-350 words), grade 1-8 reading level. Simple vocabulary, common narrative elements.
- Component location: h.1.mlp.c_fc
- Component firing rate: 0.95% (~1 in 105 tokens)

## Output tokens (what the model produces when this component fires)

**Output PMI — tokens produced at higher-than-base-rate when this component fires:**
- 'course': 4.38
- 'afar': 3.88
- 'inspiration': 3.47
- 'paper': 3.24
- 'emot': 3.23
- 'energy': 3.19
- 'betrayal': 3.16
- 'history': 3.16
- 'hope': 3.16
- 'nature': 3.13

**Output precision — of all probability mass for token X, what fraction is at positions where this component fires?**
- 'course': 76%
- 'afar': 46%
- 'inspiration': 30%
- 'paper': 24%
- 'emot': 24%
- 'energy': 23%
- 'betrayal': 22%
- 'history': 22%
- 'hope': 22%
- 'nature': 22%

## Input tokens (what causes this component to fire)

**Input PMI — tokens with higher-than-base-rate co-occurrence with firing:**
- 'about': 4.21
- 'of': 4.07
- 'from': 3.67
- 'by': 3.46
- 'outsm': 1.54
- 'cal': 1.47

**Input recall — most common tokens when the component fires:**
- 'of': 71%
- 'about': 14%
- 'from': 8%
- 'by': 4%
- 'with': 0%
- 'in': 0%
- 'for': 0%
- 'the': 0%

**Input precision — probability the component fires given the current token is X:**
- 'about': 64%
- 'of': 55%
- 'from': 37%
- 'by': 30%
- 'outsm': 4%
- 'cal': 4%
- 'fro': 4%
- '##venge': 2%

## Activation examples — where the component fires

<<delimiters>> mark tokens where this component is active.

1. people saw each other. the boy had been betrayed <<by>> a close friend, so he wished for the orb
2. tough, " she thought, feeling the heat <<of>> the sun. alice trekked through the sand
3. ##ing the magic of the earth. excited to learn <<from>> them, rita spent her days in the garden,
4. hoverboard zoomed along, and they shared stories <<about>> their lives. alex learned that the woman was an
5. , except for a shy boy who felt out <<of>> place. he watched the colorful decorations and heard
6. to finding your place. " anne felt a spark <<of>> hope. she wanted to understand the stone ' s
7. coins and old trinkets. he thought <<about>> taking some, but then he remembered the island '
8. top, he found an open window with a view <<of>> the sea. and there, sitting on the led
9. stroke brought her joy, and she felt the warmth <<of>> connection with the universe. she shared her work with
10. ##ined a new friend. this holiday would be one <<of>> joy, not sadness. next year, he would
11. peace, alex took a moment to enjoy the beauty <<of>> the garden instead. he picked a single fruit,
12. by the river held a yearly festival. people <<from>> all around would come to dance, sing, and
13. with it, a fearsome dragon. the people <<of>> the kingdom were scared. the king sent out a
14. kindness and tradition. kim felt grateful to be part <<of>> something so special. so, dear reader, remember
15. children, laughter mixing with the sun. the map <<of>> sorrow became a map <<of>> hope. leo learned that
16. in silence, waiting for a plan. he thought <<about>> his past
17. ! " she shouted. the elephant responded <<by>> doing a little dance, flapping its ears and
18. watched the people with wide eyes. they were curious <<about>> the town below. in the town, mia and
19. he missed every shot. samuel felt frustrated and thought <<about>> quitting the game. one afternoon, he
20. to be seen, yet feared the risk <<of>> rejection. anne ' s words flowed onto
21. the love we share. she smiled as she thought <<of>> her friend, knowing that together, they could face
22. his shell and said, " let ' s think <<of>> a plan. " the frogs looked at each other
23. wait. his phone buzzed. it was a message <<from>> his sister, maria. " i ' ll be
24. the animals still did not notice. with each beat <<of>> its wings, the owl felt more invisible. "
25. rusted locket. inside, there was a picture <<of>> a young woman who looked just like her. curious
26. . her mind raced, and she felt the weight <<of>> a thousand stories pressing down on her
27. could take me to space? " with a spark <<of>> imagination, he walked up to the robot. "
28. what do you want, lass? " one <<of>> them asked, scratching his beard. she
29. " the tree gave you love. you are part <<of>> this vine. " the apple thought hard
30. connection. he sent it off, feeling a sense <<of>> peace for the first time in weeks. as autumn

## Activation examples — what the model produces

Same examples with <<delimiters>> shifted right by one — showing the token that follows each firing position.

1. people saw each other. the boy had been betrayed by <<a>> close friend, so he wished for the orb
2. tough, " she thought, feeling the heat of <<the>> sun. alice trekked through the sand
3. ##ing the magic of the earth. excited to learn from <<them>>, rita spent her days in the garden,
4. hoverboard zoomed along, and they shared stories about <<their>> lives. alex learned that the woman was an
5. , except for a shy boy who felt out of <<place>>. he watched the colorful decorations and heard
6. to finding your place. " anne felt a spark of <<hope>>. she wanted to understand the stone ' s
7. coins and old trinkets. he thought about <<taking>> some, but then he remembered the island '
8. top, he found an open window with a view of <<the>> sea. and there, sitting on the led
9. stroke brought her joy, and she felt the warmth of <<connection>> with the universe. she shared her work with
10. ##ined a new friend. this holiday would be one of <<joy>>, not sadness. next year, he would
11. peace, alex took a moment to enjoy the beauty of <<the>> garden instead. he picked a single fruit,
12. by the river held a yearly festival. people from <<all>> around would come to dance, sing, and
13. with it, a fearsome dragon. the people of <<the>> kingdom were scared. the king sent out a
14. kindness and tradition. kim felt grateful to be part of <<something>> so special. so, dear reader, remember
15. children, laughter mixing with the sun. the map of <<sorrow>> became a map of <<hope>>. leo learned that
16. in silence, waiting for a plan. he thought about <<his>> past
17. ! " she shouted. the elephant responded by <<doing>> a little dance, flapping its ears and
18. watched the people with wide eyes. they were curious about <<the>> town below. in the town, mia and
19. he missed every shot. samuel felt frustrated and thought about <<qu>>itting the game. one afternoon, he
20. to be seen, yet feared the risk of <<re>>jection. anne ' s words flowed onto
21. the love we share. she smiled as she thought of <<her>> friend, knowing that together, they could face
22. his shell and said, " let ' s think of <<a>> plan. " the frogs looked at each other
23. wait. his phone buzzed. it was a message from <<his>> sister, maria. " i ' ll be
24. the animals still did not notice. with each beat of <<its>> wings, the owl felt more invisible. "
25. rusted locket. inside, there was a picture of <<a>> young woman who looked just like her. curious
26. . her mind raced, and she felt the weight of <<a>> thousand stories pressing down on her
27. could take me to space? " with a spark of <<imagination>>, he walked up to the robot. "
28. what do you want, lass? " one of <<them>> asked, scratching his beard. she
29. " the tree gave you love. you are part of <<this>> vine. " the apple thought hard
30. connection. he sent it off, feeling a sense of <<peace>> for the first time in weeks. as autumn


## Task

Give a 8-word-or-fewer label describing this component's function. The label should read like a short description of the job this component does in the network. Use both the input and output evidence.

Examples of good labels across different component types:
- "word stem completion (stems → suffixes)"
- "closes dialogue with quotation marks"
- "object pronouns after verbs"
- "story-ending moral resolution vocabulary"
- "aquatic scene vocabulary (frog, river, pond)"
- "'of course' and abstract nouns after prepositions"

Say "unclear" if the evidence is too weak or diffuse. FORBIDDEN vague words: narrative, story, character, theme, descriptive, content, transition, scene. Lowercase only.
