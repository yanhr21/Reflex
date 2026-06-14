# Cosmos3 Live Query Training Coverage Audit

- train rows: `2899`
- live Cosmos queries: `173`
- train role/mode mismatch: `1193` (`0.4115`)
- undercovered live queries: `74` (`0.4277`)
- prefix tolerance: `16`
- rel L2 tolerance: `0.05`
- rel yz tolerance: `0.03`

## Live Query Roles

- `peg_recovery`: `8`
- `target_motion_observed`: `30`
- `target_post_motion`: `135`

## Training Prefix Roles

- `insert_resume`: `733`
- `peg_recovery`: `127`
- `static_late_monitor`: `160`
- `static_monitor`: `160`
- `target_motion_observed`: `573`
- `target_post_motion`: `573`
- `target_pre_motion`: `573`

## Undercovered Queries

- `hole_late_constant` iter `12` frame `190` role `target_post_motion` rel `[-0.09290620684623718, -0.04245612025260925, -0.02145497500896454]` nearest_consistent_l2 `0.045313701033592224`
- `hole_late_constant` iter `13` frame `198` role `target_post_motion` rel `[-0.09328937530517578, -0.042116761207580566, -0.02477140724658966]` nearest_consistent_l2 `0.0568113774061203`
- `hole_late_constant` iter `14` frame `206` role `target_post_motion` rel `[-0.09375181794166565, -0.041634559631347656, -0.02832375466823578]` nearest_consistent_l2 `0.05850354954600334`
- `hole_late_constant` iter `15` frame `214` role `target_post_motion` rel `[-0.09434273838996887, -0.04093955457210541, -0.03022725135087967]` nearest_consistent_l2 `0.05864311754703522`
- `hole_late_constant` iter `16` frame `222` role `target_post_motion` rel `[-0.0949823260307312, -0.040505409240722656, -0.03241948038339615]` nearest_consistent_l2 `0.05919494107365608`
- `hole_late_constant` iter `17` frame `230` role `target_post_motion` rel `[-0.09551858901977539, -0.03945903480052948, -0.03381577879190445]` nearest_consistent_l2 `0.05896003916859627`
- `hole_late_constant` iter `18` frame `238` role `target_post_motion` rel `[-0.0962771475315094, -0.039167165756225586, -0.03858635574579239]` nearest_consistent_l2 `0.06126838177442551`
- `hole_late_constant` iter `19` frame `246` role `target_post_motion` rel `[-0.09722325205802917, -0.03827497363090515, -0.04492602497339249]` nearest_consistent_l2 `0.06444476544857025`
- `hole_late_constant` iter `20` frame `254` role `target_post_motion` rel `[-0.09782972931861877, -0.037406742572784424, -0.05080467462539673]` nearest_consistent_l2 `0.0678553506731987`
- `hole_late_constant` iter `21` frame `262` role `target_post_motion` rel `[-0.0982719361782074, -0.036696985363960266, -0.05661169812083244]` nearest_consistent_l2 `0.07167975604534149`
- `hole_late_constant` iter `22` frame `270` role `target_post_motion` rel `[-0.09877732396125793, -0.03599253296852112, -0.0624958798289299]` nearest_consistent_l2 `0.07581698149442673`
- `hole_late_constant` iter `23` frame `278` role `target_post_motion` rel `[-0.09944179654121399, -0.0344146192073822, -0.06671258807182312]` nearest_consistent_l2 `0.07831104099750519`
- `hole_late_constant` iter `24` frame `286` role `target_post_motion` rel `[-0.10016724467277527, -0.03450655937194824, -0.07151314616203308]` nearest_consistent_l2 `0.08227154612541199`
- `hole_late_constant` iter `25` frame `294` role `target_post_motion` rel `[-0.10154452919960022, -0.03810565173625946, -0.07551456242799759]` nearest_consistent_l2 `0.08721281588077545`
- `hole_late_fast_shift` iter `7` frame `142` role `target_post_motion` rel `[-0.0021722912788391113, 0.003340408205986023, 0.003201432526111603]` nearest_consistent_l2 `0.054762937128543854`
- `hole_late_sine` iter `16` frame `208` role `target_post_motion` rel `[-0.1067926287651062, -0.028236746788024902, -0.02508952096104622]` nearest_consistent_l2 `0.04214724525809288`
- `hole_late_sine` iter `17` frame `216` role `target_post_motion` rel `[-0.10763406753540039, -0.031731024384498596, -0.0330253429710865]` nearest_consistent_l2 `0.049504391849040985`
- `hole_late_sine` iter `18` frame `224` role `target_post_motion` rel `[-0.10821831226348877, -0.03251354396343231, -0.03566407039761543]` nearest_consistent_l2 `0.05172048136591911`
- `hole_late_sine` iter `19` frame `232` role `target_post_motion` rel `[-0.10920926928520203, -0.036041200160980225, -0.04031972959637642]` nearest_consistent_l2 `0.057298000901937485`
- `hole_late_sine` iter `20` frame `240` role `target_post_motion` rel `[-0.11018258333206177, -0.036977291107177734, -0.04308941960334778]` nearest_consistent_l2 `0.059782590717077255`
- `hole_late_sine` iter `21` frame `248` role `target_post_motion` rel `[-0.11136013269424438, -0.037230730056762695, -0.046819522976875305]` nearest_consistent_l2 `0.06253053992986679`
- `hole_late_sine` iter `22` frame `256` role `target_post_motion` rel `[-0.1128770112991333, -0.03892955183982849, -0.050285518169403076]` nearest_consistent_l2 `0.06614252924919128`
- `hole_late_sine` iter `23` frame `264` role `target_post_motion` rel `[-0.11522704362869263, -0.04458510875701904, -0.05273505300283432]` nearest_consistent_l2 `0.07173804193735123`
- `hole_late_sine` iter `24` frame `272` role `target_post_motion` rel `[-0.11761945486068726, -0.049646228551864624, -0.05265418812632561]` nearest_consistent_l2 `0.0753171369433403`
- `hole_late_sine` iter `25` frame `280` role `target_post_motion` rel `[-0.11965471506118774, -0.05193844437599182, -0.052371762692928314]` nearest_consistent_l2 `0.07689686864614487`
- `hole_late_sine` iter `26` frame `288` role `target_post_motion` rel `[-0.12152254581451416, -0.057352662086486816, -0.05208580568432808]` nearest_consistent_l2 `0.08094433695077896`
- `hole_late_sine` iter `27` frame `296` role `target_post_motion` rel `[-0.12327516078948975, -0.0606914758682251, -0.05130045861005783]` nearest_consistent_l2 `0.08322347700595856`
- `hole_late_sine` iter `4` frame `128` role `target_post_motion` rel `[-0.2078912854194641, -0.11515002697706223, -0.007841065526008606]` nearest_consistent_l2 `0.07727475464344025`
- `hole_late_sine` iter `5` frame `136` role `target_post_motion` rel `[-0.1493813395500183, -0.06843531131744385, -0.006120063364505768]` nearest_consistent_l2 `0.05188145488500595`
- `hole_late_sine` iter `12` frame `192` role `target_post_motion` rel `[-0.10251826047897339, -0.03482988476753235, -0.02816276252269745]` nearest_consistent_l2 `0.04794396460056305`
- `hole_late_sine` iter `13` frame `200` role `target_post_motion` rel `[-0.10303384065628052, -0.03380313515663147, -0.0293295755982399]` nearest_consistent_l2 `0.04969324171543121`
- `hole_late_sine` iter `14` frame `208` role `target_post_motion` rel `[-0.10378497838973999, -0.03315025568008423, -0.03101062774658203]` nearest_consistent_l2 `0.050106897950172424`
- `hole_late_sine` iter `15` frame `216` role `target_post_motion` rel `[-0.10448560118675232, -0.033221036195755005, -0.03234802931547165]` nearest_consistent_l2 `0.050807561725378036`
- `hole_late_sine` iter `16` frame `224` role `target_post_motion` rel `[-0.10498660802841187, -0.0344223827123642, -0.03312788903713226]` nearest_consistent_l2 `0.052093200385570526`
- `hole_late_sine` iter `17` frame `232` role `target_post_motion` rel `[-0.10518652200698853, -0.035715386271476746, -0.034162841737270355]` nearest_consistent_l2 `0.053675733506679535`
- `hole_late_sine` iter `18` frame `240` role `target_post_motion` rel `[-0.10583826899528503, -0.035591259598731995, -0.03459085524082184]` nearest_consistent_l2 `0.05372166633605957`
- `hole_late_sine` iter `19` frame `248` role `target_post_motion` rel `[-0.10609400272369385, -0.034593239426612854, -0.03276239335536957]` nearest_consistent_l2 `0.051784928888082504`
- `hole_late_sine` iter `20` frame `256` role `target_post_motion` rel `[-0.10611975193023682, -0.03397811949253082, -0.030469253659248352]` nearest_consistent_l2 `0.04993079602718353`
- `hole_late_sine` iter `21` frame `264` role `target_post_motion` rel `[-0.10650724172592163, -0.03727144002914429, -0.03330002725124359]` nearest_consistent_l2 `0.05412261560559273`
- `hole_late_sine` iter `22` frame `272` role `target_post_motion` rel `[-0.10743361711502075, -0.04551011323928833, -0.03926635533571243]` nearest_consistent_l2 `0.06404340267181396`
- `hole_late_sine` iter `23` frame `280` role `target_post_motion` rel `[-0.10914823412895203, -0.06075035035610199, -0.05073253810405731]` nearest_consistent_l2 `0.08283988386392593`
- `hole_late_sine` iter `24` frame `288` role `target_post_motion` rel `[-0.09497806429862976, -0.1050877720117569, -0.05202018469572067]` nearest_consistent_l2 `0.12330905348062515`
- `hole_late_sine` iter `25` frame `296` role `target_post_motion` rel `[-0.08598732948303223, -0.12095482647418976, -0.04423413425683975]` nearest_consistent_l2 `0.13669151067733765`
- `hole_late_continuous_insert` iter `13` frame `202` role `target_post_motion` rel `[-0.11371510475873947, -0.033370111137628555, -0.010689027607440948]` nearest_consistent_l2 `0.04513649642467499`
- `hole_late_continuous_insert` iter `14` frame `210` role `target_post_motion` rel `[-0.11449994146823883, -0.03566960245370865, -0.013953104615211487]` nearest_consistent_l2 `0.04296567291021347`
- `hole_late_continuous_insert` iter `15` frame `218` role `target_post_motion` rel `[-0.11648410558700562, -0.04305104911327362, -0.01603185385465622]` nearest_consistent_l2 `0.05062207207083702`
- `hole_late_continuous_insert` iter `16` frame `226` role `target_post_motion` rel `[-0.11732380092144012, -0.04557356983423233, -0.01902683824300766]` nearest_consistent_l2 `0.053970493376255035`
- `hole_late_continuous_insert` iter `17` frame `234` role `target_post_motion` rel `[-0.11750760674476624, -0.043184444308280945, -0.016378410160541534]` nearest_consistent_l2 `0.05087817832827568`
- `hole_late_continuous_insert` iter `18` frame `242` role `target_post_motion` rel `[-0.1174054890871048, -0.04193329066038132, -0.012977413833141327]` nearest_consistent_l2 `0.048753418028354645`
- `hole_late_continuous_insert` iter `19` frame `250` role `target_post_motion` rel `[-0.11733581125736237, -0.041704148054122925, -0.010554403066635132]` nearest_consistent_l2 `0.04800008609890938`
- `hole_late_continuous_insert` iter `20` frame `258` role `target_post_motion` rel `[-0.11723947525024414, -0.041260384023189545, -0.008433453738689423]` nearest_consistent_l2 `0.047192126512527466`
- `hole_late_continuous_insert` iter `21` frame `266` role `target_post_motion` rel `[-0.11716160178184509, -0.03993452340364456, -0.007747001945972443]` nearest_consistent_l2 `0.04577771574258804`
- `hole_late_continuous_insert` iter `22` frame `274` role `target_post_motion` rel `[-0.11692557483911514, -0.038632627576589584, -0.007220752537250519]` nearest_consistent_l2 `0.044408947229385376`
- `hole_late_continuous_insert` iter `23` frame `282` role `target_post_motion` rel `[-0.11709636449813843, -0.03908903896808624, -0.0075669065117836]` nearest_consistent_l2 `0.044913604855537415`
- `hole_late_continuous_insert` iter `24` frame `290` role `target_post_motion` rel `[-0.11709930002689362, -0.038787566125392914, -0.008225850760936737]` nearest_consistent_l2 `0.044713687151670456`
- `hole_late_continuous_insert` iter `25` frame `298` role `target_post_motion` rel `[-0.1170763447880745, -0.037806205451488495, -0.009300418198108673]` nearest_consistent_l2 `0.04392898082733154`
- `hole_late_move_stop` iter `17` frame `223` role `target_post_motion` rel `[-0.0938485860824585, -0.011202141642570496, -0.003006458282470703]` nearest_consistent_l2 `0.027458513155579567`
- `hole_late_move_stop` iter `20` frame `247` role `target_post_motion` rel `[-0.09232041239738464, -0.010347649455070496, -0.002828367054462433]` nearest_consistent_l2 `0.028193552047014236`
- `hole_late_move_stop` iter `21` frame `255` role `target_post_motion` rel `[-0.09304386377334595, -0.012403666973114014, -0.0028712525963783264]` nearest_consistent_l2 `0.028818843886256218`
- `hole_late_move_stop` iter `22` frame `263` role `target_post_motion` rel `[-0.09389954805374146, -0.014781072735786438, -0.00292418897151947]` nearest_consistent_l2 `0.029711514711380005`
- `hole_late_move_stop` iter `23` frame `271` role `target_post_motion` rel `[-0.09449636936187744, -0.016388729214668274, -0.002957545220851898]` nearest_consistent_l2 `0.03040667623281479`
- `hole_late_move_stop` iter `24` frame `279` role `target_post_motion` rel `[-0.09500741958618164, -0.018109604716300964, -0.0029813945293426514]` nearest_consistent_l2 `0.031321682035923004`
- `hole_late_move_stop` iter `25` frame `287` role `target_post_motion` rel `[-0.09477749466896057, -0.019184723496437073, -0.00296667218208313]` nearest_consistent_l2 `0.032282568514347076`
- `hole_late_move_stop` iter `26` frame `295` role `target_post_motion` rel `[-0.09580093622207642, -0.020828694105148315, -0.00301457941532135]` nearest_consistent_l2 `0.03292372077703476`
- `hole_late_fast_shift` iter `19` frame `224` role `target_post_motion` rel `[-0.07486513257026672, -0.0024075955152511597, 0.0045033469796180725]` nearest_consistent_l2 `0.018478581681847572`
- `hole_late_fast_shift` iter `20` frame `232` role `peg_recovery` rel `[-0.07495740056037903, -0.002324119210243225, 0.004453904926776886]` nearest_consistent_l2 `0.11676624417304993`
- `hole_late_fast_shift` iter `21` frame `240` role `peg_recovery` rel `[-0.07512101531028748, -0.0022166669368743896, 0.004389382898807526]` nearest_consistent_l2 `0.11661294847726822`
- `hole_late_fast_shift` iter `22` frame `248` role `peg_recovery` rel `[-0.07491475343704224, -0.0021843016147613525, 0.00430385023355484]` nearest_consistent_l2 `0.11673028022050858`
- `hole_late_fast_shift` iter `23` frame `256` role `peg_recovery` rel `[-0.07514640688896179, -0.0021215081214904785, 0.0042586252093315125]` nearest_consistent_l2 `0.11652835458517075`
- `hole_late_fast_shift` iter `24` frame `264` role `peg_recovery` rel `[-0.07571351528167725, -0.002009093761444092, 0.004221521317958832]` nearest_consistent_l2 `0.11607273668050766`
- `hole_late_fast_shift` iter `25` frame `272` role `peg_recovery` rel `[-0.07636764645576477, -0.0019404441118240356, 0.0042167529463768005]` nearest_consistent_l2 `0.11556265503168106`
- `hole_late_fast_shift` iter `26` frame `280` role `peg_recovery` rel `[-0.07750019431114197, -0.0018617361783981323, 0.00423198938369751]` nearest_consistent_l2 `0.11469145864248276`
- `hole_late_fast_shift` iter `27` frame `288` role `peg_recovery` rel `[-0.07872268557548523, -0.0017901957035064697, 0.004239201545715332]` nearest_consistent_l2 `0.11374875158071518`
- `hole_late_fast_shift` iter `28` frame `296` role `target_post_motion` rel `[-0.08025449514389038, -0.0016613155603408813, 0.004281237721443176]` nearest_consistent_l2 `0.0365351065993309`
