Optimization Results - 20250129_023617

Configuration:
- Seeds: [41, 38, 37, 36, 35]
- Metrics: ['Common Crawl Cross Entropy', 'C4 Cross Entropy', 'Wikipedia Cross Entropy', 'Stack Exchange Cross Entropy', 'Github Cross Entropy', 'ArXiv Cross Entropy', 'Book Cross Entropy', 'Hellaswag Accuracy', 'PIQA Accuracy', 'ARC Easy Accuracy']
- Optimizers: ['SMAC', 'RandomSearch', 'GridSearch']

Directory Structure:
- data/: Contains the main results pickle file
- logs/: Contains individual optimization logs
- plots/: Contains interactive visualization plots

Results Summary:

Common Crawl Cross Entropy:
- SMAC:
  Mean best value: 2.2797 ± 0.0033
  Overall best value: 2.2759
  Best config: [-2.9912355436513, 0.1600147874145, -2.787276892761, -0.1708009169056, 2.9994138913856]
- RandomSearch:
  Mean best value: 2.2939 ± 0.0117
  Overall best value: 2.2796
  Best config: [-2.8888034342754323, -0.6748869001637186, -2.931429183126238, -1.0781428875223986, 2.1999026375296644]
- GridSearch:
  Mean best value: 2.2898 ± 0.0124
  Overall best value: 2.2775
  Best config: [-3.0, 0.0, -3.0, -3.0, 3.0]

C4 Cross Entropy:
- SMAC:
  Mean best value: 1.3195 ± 0.0014
  Overall best value: 1.3176
  Best config: [-2.9893640151162, 0.1536280428717, -2.9641922394796, -0.1250002805563, 2.8978059248319]
- RandomSearch:
  Mean best value: 1.3273 ± 0.0062
  Overall best value: 1.3196
  Best config: [-2.8888034342754323, -0.6748869001637186, -2.931429183126238, -1.0781428875223986, 2.1999026375296644]
- GridSearch:
  Mean best value: 1.3255 ± 0.0068
  Overall best value: 1.3186
  Best config: [-3.0, 0.0, -3.0, -3.0, 3.0]

Wikipedia Cross Entropy:
- SMAC:
  Mean best value: 0.8584 ± 0.0003
  Overall best value: 0.8582
  Best config: [2.9947926090212, -2.9978165631321, -2.9885113749529, -2.9648163639318, -2.7757959493572]
- RandomSearch:
  Mean best value: 0.8753 ± 0.0072
  Overall best value: 0.8683
  Best config: [2.483937062090109, -1.869033806310948, -1.786626121304621, -1.3013977578750622, -2.2148821614503156]
- GridSearch:
  Mean best value: 0.8635 ± 0.0073
  Overall best value: 0.8581
  Best config: [3.0, -3.0, -3.0, -3.0, -3.0]

Stack Exchange Cross Entropy:
- SMAC:
  Mean best value: 0.8081 ± 0.0004
  Overall best value: 0.8075
  Best config: [-2.5921526976706, 2.9831871733146, 1.5947828521841, -2.8452744856905, -2.8120962480526]
- RandomSearch:
  Mean best value: 0.8194 ± 0.0034
  Overall best value: 0.8149
  Best config: [-0.9483854859874867, 2.9890698942292158, -1.6053786038078337, -2.847498729169347, -2.095602573847712]
- GridSearch:
  Mean best value: 0.8089 ± 0.0008
  Overall best value: 0.8085
  Best config: [-3.0, 3.0, 0.0, -3.0, -3.0]

Github Cross Entropy:
- SMAC:
  Mean best value: 0.5359 ± 0.0003
  Overall best value: 0.5353
  Best config: [-2.9745882594676, -2.7779714920537, 2.9713061056725, -2.9650015258716, -1.2845413404656]
- RandomSearch:
  Mean best value: 0.5446 ± 0.0064
  Overall best value: 0.5377
  Best config: [-2.394080974621154, -1.2282948201952781, 2.8177921251868323, -1.6907682573650071, -2.5295724873144296]
- GridSearch:
  Mean best value: 0.5426 ± 0.0040
  Overall best value: 0.5392
  Best config: [-3.0, 0.0, 3.0, -3.0, -3.0]

ArXiv Cross Entropy:
- SMAC:
  Mean best value: 1.3157 ± 0.0002
  Overall best value: 1.3155
  Best config: [-2.999814790533, -2.9784255980796, -2.9789714034048, 2.993422216207, -2.9777811188797]
- RandomSearch:
  Mean best value: 1.3453 ± 0.0144
  Overall best value: 1.3242
  Best config: [-2.4738669037611922, -0.6753280903568593, -2.627225099847874, 2.9632364948576804, -1.8581227104001463]
- GridSearch:
  Mean best value: 1.3181 ± 0.0053
  Overall best value: 1.3155
  Best config: [-3.0, -3.0, -3.0, 3.0, -3.0]

Book Cross Entropy:
- SMAC:
  Mean best value: 2.2707 ± 0.0003
  Overall best value: 2.2703
  Best config: [-2.9985237093574, -2.9993525341339, -2.9966546133557, -2.994404757877, 2.9996490227235]
- RandomSearch:
  Mean best value: 2.3013 ± 0.0169
  Overall best value: 2.2787
  Best config: [-2.5540718271134484, -2.939706724258233, -2.4338944621928156, -1.1722536214045927, 2.7293310643961135]
- GridSearch:
  Mean best value: 2.2846 ± 0.0152
  Overall best value: 2.2703
  Best config: [-3.0, -3.0, -3.0, -3.0, 3.0]

Hellaswag Accuracy:
- SMAC:
  Mean best value: -0.2534 ± 0.0002
  Overall best value: -0.2538
  Best config: [-0.3599528580484, 0.6950700978651, 2.0265733098275, -0.3310581601229, 1.8468999627639]
- RandomSearch:
  Mean best value: -0.2948 ± 0.0004
  Overall best value: -0.2953
  Best config: [1.3480326174684585, -2.4190641816261076, 1.4242410613254926, -1.692470285947337, -2.19858710788555]
- GridSearch:
  Mean best value: -0.2945 ± 0.0002
  Overall best value: -0.2947
  Best config: [0.0, -3.0, 3.0, 0.0, -3.0]

PIQA Accuracy:
- SMAC:
  Mean best value: -0.5099 ± 0.0003
  Overall best value: -0.5102
  Best config: [1.7136711601772, -2.3972862723384, 2.9361656692144, 1.3052025818634, -1.1140575408296]
- RandomSearch:
  Mean best value: -0.5995 ± 0.0013
  Overall best value: -0.6017
  Best config: [1.71478197367043, -2.199577332903727, 2.5891029750010413, 0.6428630195805378, -2.8503154967792743]
- GridSearch:
  Mean best value: -0.5975 ± 0.0005
  Overall best value: -0.5982
  Best config: [3.0, 0.0, -3.0, 0.0, -3.0]

ARC Easy Accuracy:
- SMAC:
  Mean best value: -0.2768 ± 0.0002
  Overall best value: -0.2771
  Best config: [-2.8748941342468, -1.2724886569137, 1.4503200863458, 1.6115000806829, -2.2049563708258]
- RandomSearch:
  Mean best value: -0.4080 ± 0.0020
  Overall best value: -0.4114
  Best config: [-2.5659000559409533, 1.8328941567972326, 2.57327031242807, -1.663565597822434, -2.5118247633446367]
- GridSearch:
  Mean best value: -0.4053 ± 0.0019
  Overall best value: -0.4091
  Best config: [0.0, -3.0, 3.0, 0.0, 0.0]
