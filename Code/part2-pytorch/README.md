# CIFAR-10 Challenge -- CS-4644DL/7643 Spring 2022

1. Download [test_images.zip](https://www.dropbox.com/sh/c3nq34qe17qkln7/AADANWqMNRt1rjydqtz9S5WDa?dl=0) and put it under `./data`
2. Unzip `./data/test_images.zip` to get `./data/test_images.npy`.
3. Implement your model in `./models/my_model.py` under the `MyModel` class. Train and store your model weights. You can use your HW2 code for this if you want.
4. Run `python test.py --model /path/to/model/checkpoint/`. Append `--help` to see all options.
5. This generates `predictions.csv, which has the class that your model predicted for each image. 
6. Go to the [challenge website on EvalAI](https://eval.ai/web/challenges/challenge-page/1592/overview) and sign up using your gatech.edu email address. Submit `predictions.csv` there and see your result on the leaderboard!

Note: there will be an upper limit on the number of leadberboard submissions to prevent overfitting to the challenge test-set -- you can view this on the website before submitting.
