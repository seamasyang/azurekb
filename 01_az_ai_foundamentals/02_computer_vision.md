# fundamentals of computer vision

computer vision is one of the core areas of ai, and foucs on creating solutions enable ai application to "see" the world and make sense of it. 

## images and image processing

Images as pixel arrays (像素数组)
ignored

## machine learning for computer vision

## azure ai version

## exercise - analyze images in vision studio

## knowledge check

# fundamentals of facial recognition

## understand facial analysis

## get started with facial analysis on az

## exercise - defect faces in vision studio

# fundamentals of optical character recognition

what: machine can read the text in image using optical character recognition (OCR), the capability for ai to process word in images into machine-readable text. 

how: detect text in images -> convert it into a text-based data format -> stored/printed/other processing or analysis
why: improve speed and efficiency of work by removing the need for manual data entry
use cases: note taking, digitizing medical records, scanning checks for bank deposits, etc

## get started with az ai vision

computer system process written and printed text = intersects (computer vision and NLP)

- computer vsion: "read" the text
- nlp: make sense if readed text
OCR is the foundation of procesing text in image, using machine trained learning models to recognize individual shapes (letters, numerals, punctuation, or other elements of text)
early work on postal service to automatic sorting of mail based on postal codes

### az ai vision's OCR engine

AZ AI Vision read API (read OCR engine) to extract text from images, PDFS, and TIFF files. 
read OCR engine use latest recognition models and is optimized for images; consider the number of lines of text, images that include text, and handwriting
how: OCR engine takes in an image file and identifies bounding boxes, or coordinates, where items are located within an image. 
return result:

- pages, one for each page of text, including info about the page size and orientation
- lines, the lines of text on a page
- words, the words in a line of text, including the bounding box coordinates and text itself

## get started with vision studio on az

required az resource type

- az ai vision: use it if don't intent to use any other ai services, or want track utilization and costs for ai vision resource separately
- az zi service: general resource about az ai services. 
ways to use az ai vision read api
- vision studio
- rest api
- sdk

when open vision studio, default resource must be an az ai service resource, rather than az ai vision resource
![sample](../imgs/vision_ocr_sample.jpg)
```json
[
  {
    "lines": [
      {
        "text": "general",
        "boundingPolygon": [
          {
            "x": 474,
            "y": 190
          },
          {
            "x": 1005,
            "y": 149
          },
          {
            "x": 1010,
            "y": 275
          },
          {
            "x": 476,
            "y": 313
          }
        ],
        "words": [
          {
            "text": "general",
            "boundingPolygon": [
              {
                "x": 478,
                "y": 190
              },
              {
                "x": 1002,
                "y": 150
              },
              {
                "x": 1006,
                "y": 278
              },
              {
                "x": 480,
                "y": 313
              }
            ],
            "confidence": 0.941
          }
        ]
      },
      {
        "text": "train process",
        "boundingPolygon": [
          {
            "x": 1191,
            "y": 107
          },
          {
            "x": 2356,
            "y": 168
          },
          {
            "x": 2353,
            "y": 304
          },
          {
            "x": 1191,
            "y": 267
          }
        ],
        "words": [
          {
            "text": "train",
            "boundingPolygon": [
              {
                "x": 1215,
                "y": 109
              },
              {
                "x": 1618,
                "y": 128
              },
              {
                "x": 1620,
                "y": 285
              },
              {
                "x": 1218,
                "y": 271
              }
            ],
            "confidence": 0.851
          },
          {
            "text": "process",
            "boundingPolygon": [
              {
                "x": 1858,
                "y": 145
              },
              {
                "x": 2344,
                "y": 195
              },
              {
                "x": 2346,
                "y": 304
              },
              {
                "x": 1861,
                "y": 292
              }
            ],
            "confidence": 0.972
          }
        ]
      },
      {
        "text": "1. define a loss function",
        "boundingPolygon": [
          {
            "x": 706,
            "y": 403
          },
          {
            "x": 2691,
            "y": 402
          },
          {
            "x": 2691,
            "y": 550
          },
          {
            "x": 706,
            "y": 553
          }
        ],
        "words": [
          {
            "text": "1.",
            "boundingPolygon": [
              {
                "x": 706,
                "y": 405
              },
              {
                "x": 845,
                "y": 405
              },
              {
                "x": 847,
                "y": 553
              },
              {
                "x": 708,
                "y": 554
              }
            ],
            "confidence": 0.955
          },
          {
            "text": "define",
            "boundingPolygon": [
              {
                "x": 932,
                "y": 404
              },
              {
                "x": 1362,
                "y": 403
              },
              {
                "x": 1364,
                "y": 544
              },
              {
                "x": 935,
                "y": 551
              }
            ],
            "confidence": 0.771
          },
          {
            "text": "a",
            "boundingPolygon": [
              {
                "x": 1468,
                "y": 403
              },
              {
                "x": 1547,
                "y": 403
              },
              {
                "x": 1548,
                "y": 542
              },
              {
                "x": 1470,
                "y": 543
              }
            ],
            "confidence": 0.965
          },
          {
            "text": "loss",
            "boundingPolygon": [
              {
                "x": 1672,
                "y": 403
              },
              {
                "x": 1917,
                "y": 403
              },
              {
                "x": 1917,
                "y": 540
              },
              {
                "x": 1673,
                "y": 541
              }
            ],
            "confidence": 0.938
          },
          {
            "text": "function",
            "boundingPolygon": [
              {
                "x": 2060,
                "y": 404
              },
              {
                "x": 2674,
                "y": 408
              },
              {
                "x": 2674,
                "y": 547
              },
              {
                "x": 2061,
                "y": 541
              }
            ],
            "confidence": 0.566
          }
        ]
      },
      {
        "text": "2. introduce training data",
        "boundingPolygon": [
          {
            "x": 708,
            "y": 672
          },
          {
            "x": 2848,
            "y": 649
          },
          {
            "x": 2850,
            "y": 780
          },
          {
            "x": 710,
            "y": 822
          }
        ],
        "words": [
          {
            "text": "2.",
            "boundingPolygon": [
              {
                "x": 728,
                "y": 693
              },
              {
                "x": 925,
                "y": 682
              },
              {
                "x": 927,
                "y": 810
              },
              {
                "x": 731,
                "y": 823
              }
            ],
            "confidence": 0.897
          },
          {
            "text": "introduce",
            "boundingPolygon": [
              {
                "x": 1001,
                "y": 677
              },
              {
                "x": 1678,
                "y": 654
              },
              {
                "x": 1679,
                "y": 777
              },
              {
                "x": 1003,
                "y": 806
              }
            ],
            "confidence": 0.868
          },
          {
            "text": "training",
            "boundingPolygon": [
              {
                "x": 1882,
                "y": 650
              },
              {
                "x": 2400,
                "y": 651
              },
              {
                "x": 2400,
                "y": 769
              },
              {
                "x": 1884,
                "y": 772
              }
            ],
            "confidence": 0.586
          },
          {
            "text": "data",
            "boundingPolygon": [
              {
                "x": 2556,
                "y": 653
              },
              {
                "x": 2841,
                "y": 661
              },
              {
                "x": 2841,
                "y": 775
              },
              {
                "x": 2556,
                "y": 770
              }
            ],
            "confidence": 0.938
          }
        ]
      },
      {
        "text": "2. Forward Pass",
        "boundingPolygon": [
          {
            "x": 746,
            "y": 899
          },
          {
            "x": 2038,
            "y": 871
          },
          {
            "x": 2039,
            "y": 987
          },
          {
            "x": 746,
            "y": 1022
          }
        ],
        "words": [
          {
            "text": "2.",
            "boundingPolygon": [
              {
                "x": 749,
                "y": 899
              },
              {
                "x": 865,
                "y": 898
              },
              {
                "x": 863,
                "y": 1021
              },
              {
                "x": 747,
                "y": 1022
              }
            ],
            "confidence": 0.317
          },
          {
            "text": "Forward",
            "boundingPolygon": [
              {
                "x": 1094,
                "y": 896
              },
              {
                "x": 1563,
                "y": 886
              },
              {
                "x": 1561,
                "y": 1002
              },
              {
                "x": 1092,
                "y": 1015
              }
            ],
            "confidence": 0.769
          },
          {
            "text": "Pass",
            "boundingPolygon": [
              {
                "x": 1745,
                "y": 880
              },
              {
                "x": 2028,
                "y": 871
              },
              {
                "x": 2026,
                "y": 986
              },
              {
                "x": 1743,
                "y": 996
              }
            ],
            "confidence": 0.883
          }
        ]
      },
      {
        "text": "4. Calculate loss",
        "boundingPolygon": [
          {
            "x": 729,
            "y": 1112
          },
          {
            "x": 2164,
            "y": 1097
          },
          {
            "x": 2167,
            "y": 1223
          },
          {
            "x": 729,
            "y": 1249
          }
        ],
        "words": [
          {
            "text": "4.",
            "boundingPolygon": [
              {
                "x": 732,
                "y": 1128
              },
              {
                "x": 940,
                "y": 1117
              },
              {
                "x": 938,
                "y": 1243
              },
              {
                "x": 730,
                "y": 1250
              }
            ],
            "confidence": 0.834
          },
          {
            "text": "Calculate",
            "boundingPolygon": [
              {
                "x": 1102,
                "y": 1110
              },
              {
                "x": 1689,
                "y": 1098
              },
              {
                "x": 1690,
                "y": 1227
              },
              {
                "x": 1101,
                "y": 1238
              }
            ],
            "confidence": 0.597
          },
          {
            "text": "loss",
            "boundingPolygon": [
              {
                "x": 1885,
                "y": 1098
              },
              {
                "x": 2164,
                "y": 1103
              },
              {
                "x": 2167,
                "y": 1225
              },
              {
                "x": 1887,
                "y": 1225
              }
            ],
            "confidence": 0.964
          }
        ]
      },
      {
        "text": "5. Backpropagation",
        "boundingPolygon": [
          {
            "x": 768,
            "y": 1315
          },
          {
            "x": 2195,
            "y": 1315
          },
          {
            "x": 2196,
            "y": 1447
          },
          {
            "x": 768,
            "y": 1449
          }
        ],
        "words": [
          {
            "text": "5.",
            "boundingPolygon": [
              {
                "x": 796,
                "y": 1316
              },
              {
                "x": 1016,
                "y": 1316
              },
              {
                "x": 1010,
                "y": 1450
              },
              {
                "x": 789,
                "y": 1450
              }
            ],
            "confidence": 0.856
          },
          {
            "text": "Backpropagation",
            "boundingPolygon": [
              {
                "x": 1145,
                "y": 1317
              },
              {
                "x": 2161,
                "y": 1327
              },
              {
                "x": 2158,
                "y": 1439
              },
              {
                "x": 1139,
                "y": 1450
              }
            ],
            "confidence": 0.07
          }
        ]
      },
      {
        "text": "6. update parameters",
        "boundingPolygon": [
          {
            "x": 783,
            "y": 1542
          },
          {
            "x": 2450,
            "y": 1545
          },
          {
            "x": 2449,
            "y": 1697
          },
          {
            "x": 782,
            "y": 1695
          }
        ],
        "words": [
          {
            "text": "6.",
            "boundingPolygon": [
              {
                "x": 814,
                "y": 1543
              },
              {
                "x": 937,
                "y": 1549
              },
              {
                "x": 928,
                "y": 1688
              },
              {
                "x": 804,
                "y": 1684
              }
            ],
            "confidence": 0.03
          },
          {
            "text": "update",
            "boundingPolygon": [
              {
                "x": 1136,
                "y": 1556
              },
              {
                "x": 1598,
                "y": 1567
              },
              {
                "x": 1593,
                "y": 1697
              },
              {
                "x": 1128,
                "y": 1694
              }
            ],
            "confidence": 0.656
          },
          {
            "text": "parameters",
            "boundingPolygon": [
              {
                "x": 1712,
                "y": 1568
              },
              {
                "x": 2437,
                "y": 1561
              },
              {
                "x": 2437,
                "y": 1674
              },
              {
                "x": 1708,
                "y": 1697
              }
            ],
            "confidence": 0.843
          }
        ]
      },
      {
        "text": "Repeat",
        "boundingPolygon": [
          {
            "x": 1158,
            "y": 1797
          },
          {
            "x": 1678,
            "y": 1798
          },
          {
            "x": 1677,
            "y": 1929
          },
          {
            "x": 1157,
            "y": 1925
          }
        ],
        "words": [
          {
            "text": "Repeat",
            "boundingPolygon": [
              {
                "x": 1182,
                "y": 1798
              },
              {
                "x": 1666,
                "y": 1799
              },
              {
                "x": 1663,
                "y": 1929
              },
              {
                "x": 1178,
                "y": 1919
              }
            ],
            "confidence": 0.758
          }
        ]
      }
    ]
  }
]
```

## exercise - read text in vision studio

1.create an az ai service resource

- subscription
- resource group
- region
- name (of instance)
- price tier: standard s0

2.connect az ai service resource to vision studio

- https:portal.vision.cognitive.azure.com
- sign in and same directory
- view all resources under get start with vision
- select a resource to work

3.extract text from images in the vision studio

- open vision studio at https:portal.vision.cognitive.azure.com
- select optical character recognition, and then extract text from images
- under try it out, acknowledge the resource usage policy by reading and checking the box
- upload one sample img
- review what returned (see previous sample);

4.delete resource group to clean up