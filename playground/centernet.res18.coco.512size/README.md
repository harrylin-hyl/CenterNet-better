# centernet.res18.coco.512size  
## Evaluation results for bbox:  
```  
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.001
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.003
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.000
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.000
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.002
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.001
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.012
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.021
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.025
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.004
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.015
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.035
```  
|  AP   |  AP50  |  AP75  |  APs  |  APm  |  APl  |  
|:-----:|:------:|:------:|:-----:|:-----:|:-----:|  
| 0.092 | 0.281  | 0.047  | 0.041 | 0.224 | 0.124 |
### Per-category bbox AP:  

| category      | AP    | category     | AP    | category       | AP    |  
|:--------------|:------|:-------------|:------|:---------------|:------|  
| person        | 2.255 | bicycle      | 0.000 | car            | 0.006 |  
| motorcycle    | 0.049 | airplane     | 0.368 | bus            | 0.360 |  
| train         | 0.138 | truck        | 0.000 | boat           | 0.000 |  
| traffic light | 0.101 | fire hydrant | 0.000 | stop sign      | 0.000 |  
| parking meter | 0.000 | bench        | 0.003 | bird           | 0.019 |  
| cat           | 0.260 | dog          | 0.064 | horse          | 0.095 |  
| sheep         | 0.004 | cow          | 0.026 | elephant       | 0.072 |  
| bear          | 0.156 | zebra        | 0.281 | giraffe        | 0.193 |  
| backpack      | 0.000 | umbrella     | 0.000 | handbag        | 0.000 |  
| tie           | 0.000 | suitcase     | 0.000 | frisbee        | 0.025 |  
| skis          | 0.043 | snowboard    | 0.013 | sports ball    | 0.001 |  
| kite          | 0.181 | baseball bat | 0.000 | baseball glove | 0.004 |  
| skateboard    | 0.123 | surfboard    | 0.016 | tennis racket  | 0.004 |  
| bottle        | 0.012 | wine glass   | 0.000 | cup            | 0.004 |  
| fork          | 0.000 | knife        | 0.000 | spoon          | 0.000 |  
| bowl          | 0.096 | banana       | 0.000 | apple          | 0.000 |  
| sandwich      | 0.001 | orange       | 0.000 | broccoli       | 0.001 |  
| carrot        | 0.001 | hot dog      | 0.000 | pizza          | 0.295 |  
| donut         | 0.000 | cake         | 0.001 | chair          | 0.016 |  
| couch         | 0.000 | potted plant | 0.000 | bed            | 0.328 |  
| dining table  | 1.211 | toilet       | 0.427 | tv             | 0.001 |  
| laptop        | 0.000 | mouse        | 0.000 | remote         | 0.002 |  
| keyboard      | 0.000 | cell phone   | 0.011 | microwave      | 0.000 |  
| oven          | 0.000 | toaster      | 0.000 | sink           | 0.012 |  
| refrigerator  | 0.032 | book         | 0.001 | clock          | 0.022 |  
| vase          | 0.000 | scissors     | 0.000 | teddy bear     | 0.000 |  
| hair drier    | 0.000 | toothbrush   | 0.000 |                |       |
