# MLzoomcamp capstone project 1

Dataset to be used: https://www.kaggle.com/datasets/prasunroy/natural-images/data

1. We want to do the same thing we did during lecture 8, i.e. we will implement transfer learning from pretrained model of imagenet.




## Containerization

In order to build and run prediction service in Docker container on port 8000:
    
    docker build . -t image_sorting_service
    docker run -d -p 8000:8000 image_sorting_service

The service gonna be accessible from here:
   
      localhost:8000/credit_rating_serivce/api/docs

## Cloud deployment 
