# h.1.mlp.down_proj:471

**Layer**: h.1.mlp.down_proj  
**Firing density**: 22.62% (~1 in 4 tokens)  
**Mean CI**: 0.2025

**Old label**: [high] punctuation and conjunctions  
**Old reasoning**: The component consistently activates on punctuation marks (.,;!) and coordinating/subordinating conjunctions (but, and, that, though, because). It appears to facilitate the transition between syntactic clauses or story segments, as evidenced by its high precision on logical connectors and its output prediction for the [EOS] token.

---

## Output function (what it predicts)

### Output precision
| Token | Precision |
|-------|-----------|
| [EOS] | 99% |
| wow | 99% |
| i | 98% |
| suddenly | 98% |
| oh | 98% |
| hey | 97% |
| however | 97% |
| hoping | 97% |
| realizing | 97% |
| timidly | 97% |
| ironically | 97% |
| she | 97% |

### Output PMI
| Token | PMI |
|-------|-----|
| [EOS] | 1.48 |
| wow | 1.47 |
| i | 1.46 |
| suddenly | 1.46 |
| oh | 1.46 |
| hey | 1.46 |
| however | 1.46 |
| hoping | 1.45 |
| realizing | 1.45 |
| timidly | 1.45 |
| ironically | 1.45 |
| she | 1.45 |

## Input function (what triggers it)

### Input recall
| Token | Recall |
|-------|--------|
| . | 33% |
| , | 26% |
| " | 11% |
| ! | 3% |
| but | 3% |
| and | 3% |
| as | 3% |
| that | 2% |

### Input precision
| Token | Precision |
|-------|-----------|
| because | 100% |
| [EOS] | 100% |
| ; | 100% |
| ##ough | 100% |
| : | 100% |
| b | 100% |
| though | 100% |
| when | 100% |

### Input PMI
| Token | PMI |
|-------|-----|
| ; | 1.49 |
| : | 1.49 |
| because | 1.49 |
| b | 1.49 |
| though | 1.49 |
| when | 1.49 |
| [EOS] | 1.49 |
| ##ough | 1.49 |

## Activation examples (dual view)

**1a. fires on**: above the ground<<. but then,>> something went wrong<<.>> the hoverboard began to spin and shake<<.>>  
**1b. says**: above the ground. <<but then, something>> went wrong. <<the>> hoverboard began to spin and shake.

**2a. fires on**: to witness her growth in imagination and adventure<<. [EOS]>> a storm broke out in the sky<<.>> a brave  
**2b. says**: to witness her growth in imagination and adventure. <<[EOS] a>> storm broke out in the sky. <<a>> brave

**3a. fires on**: <<, ">> the world is not always kind<<. " but>> his heart was bold<<, and>> he ignored her  
**3b. says**: , <<" the>> world is not always kind. <<" but his>> heart was bold, <<and he>> ignored her

**4a. fires on**: bright<<. but then,>> shadows moved<<, and>> he <<realized>> he was not alone<<. would>> he run<<, or>>  
**4b. says**: bright. <<but then, shadows>> moved, <<and he>> realized <<he>> was not alone. <<would he>> run, <<or>>

**5a. fires on**: next night<<,>> the boy returned to the dream world<<,>> flying high above the clouds again<<. ">> you did  
**5b. says**: next night, <<the>> boy returned to the dream world, <<flying>> high above the clouds again. <<" you>> did

**6a. fires on**: about growth<<,>> effort<<, and>> friendship<<. that>> day<<,>> she became a better runner and a better person  
**6b. says**: about growth, <<effort>>, <<and friendship>>. <<that day>>, <<she>> became a better runner and a better person

**7a. fires on**: her garden would always be alive in her heart<<. [EOS]>> between the waves of the sea<<,>> a fish named  
**7b. says**: her garden would always be alive in her heart. <<[EOS] between>> the waves of the sea, <<a>> fish named

**8a. fires on**: smiled<<, knowing>> she had found more <<than>> just gold <<;>> she found the magic of love<<.>> returning home<<,>>  
**8b. says**: smiled, <<knowing she>> had found more than <<just>> gold ; <<she>> found the magic of love. <<returning>> home,

**9a. fires on**: <<. ">> we are explorers<<, not>> just of land<<, but>> of our hearts<<, ">> she said<<,>> gazing  
**9b. says**: . <<" we>> are explorers, <<not just>> of land, <<but of>> our hearts, <<" she>> said, <<gazing>>

**10a. fires on**: girl stood in front of a big mirror at school<<.>> lily always wondered <<who>> she was inside<<.>> the mirror  
**10b. says**: girl stood in front of a big mirror at school. <<lily>> always wondered who <<she>> was inside. <<the>> mirror

**11a. fires on**: beautiful<<! ">> she exclaimed<<.>> the trees swayed gently<<, and>> the sound of birds filled the air<<.>> suddenly  
**11b. says**: beautiful! <<" she>> exclaimed. <<the>> trees swayed gently, <<and the>> sound of birds filled the air. <<suddenly>>

**12a. fires on**: bloom around him<<. " why are>> you sad? <<">> she asked<<.>> luis looked up in surprise and smiled  
**12b. says**: bloom around him. <<" why are you>> sad? " <<she>> asked. <<luis>> looked up in surprise and smiled

**13a. fires on**: <<,>> carrying the story far and wide<<,>> reminding everyone <<that>> the greatest treasures are those we create together<<. [EOS]>>  
**13b. says**: , <<carrying>> the story far and wide, <<reminding>> everyone that <<the>> greatest treasures are those we create together. <<[EOS]>>

**14a. fires on**: lily felt a little sad<<. ">> goodbye<<,>> friend<<! ">> she said<<,>> giving wobble a big hug  
**14b. says**: lily felt a little sad. <<" goodbye>>, <<friend>>! <<" she>> said, <<giving>> wobble a big hug

**15a. fires on**: <<. as>> they discovered the castle<<,>> they became friends<<.>> the ghost felt less lonely<<, and>> emmanuel felt brave  
**15b. says**: . <<as they>> discovered the castle, <<they>> became friends. <<the>> ghost felt less lonely, <<and emmanuel>> felt brave

---

## Ideal interpretation

*TODO: What should the label be? What reasoning process gets there?*
