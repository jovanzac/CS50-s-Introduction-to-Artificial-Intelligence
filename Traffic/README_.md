# Traffic

## Aim
To develop a convolutional neural network model capable of identifying basic traffic signs from an image

## Experimentation Process
1) I started by adopting the basic convolutional neural network model, demonstrated in the lecture, which I   obtained from the source scripts. This model included:
- A convolutional layer which learns 32 filters using a 3x3 kernel
- A Max-Pooling layer with a 2x2 pool size
- A hidden layer with 128 units
- And a dropout of 0.5
```
        Epoch 1/10
        497/497 [==============================] - 10s 18ms/step - loss: 5.6740 - accuracy: 0.0491
        Epoch 2/10
        497/497 [==============================] - 9s 18ms/step - loss: 3.5670 - accuracy: 0.0553
        Epoch 3/10
        497/497 [==============================] - 9s 18ms/step - loss: 3.5191 - accuracy: 0.0571
        Epoch 4/10
        497/497 [==============================] - 8s 16ms/step - loss: 3.4970 - accuracy: 0.0588
        Epoch 5/10
        497/497 [==============================] - 9s 18ms/step - loss: 3.4869 - accuracy: 0.0581
        Epoch 6/10
        497/497 [==============================] - 9s 18ms/step - loss: 3.4822 - accuracy: 0.0591
        Epoch 7/10
        497/497 [==============================] - 8s 17ms/step - loss: 3.4799 - accuracy: 0.0591
        Epoch 8/10
        497/497 [==============================] - 8s 17ms/step - loss: 3.4788 - accuracy: 0.0591
        Epoch 9/10
        497/497 [==============================] - 9s 17ms/step - loss: 3.4782 - accuracy: 0.0567
        Epoch 10/10
        497/497 [==============================] - 8s 16ms/step - loss: 3.4779 - accuracy: 0.0591
        331/331 - 2s - loss: 3.4875 - accuracy: 0.0530
```
The results certainly weren't satisfactory.

2) This time i added an additional convolutional and pooling layer respectively so as to maybe gain additional insight into the different features of the images. The convolutional layers had the same number of filters and the Max-Pooling layer had the same pool size as before
```
        Epoch 1/10
        497/497 [==============================] - 11s 20ms/step - loss: 4.2053 - accuracy: 0.0976
        Epoch 2/10
        497/497 [==============================] - 10s 20ms/step - loss: 2.7459 - accuracy: 0.2455
        Epoch 3/10
        497/497 [==============================] - 10s 20ms/step - loss: 2.1852 - accuracy: 0.3575
        Epoch 4/10
        497/497 [==============================] - 10s 20ms/step - loss: 1.7624 - accuracy: 0.4629
        Epoch 5/10
        497/497 [==============================] - 10s 19ms/step - loss: 1.3996 - accuracy: 0.5571
        Epoch 6/10
        497/497 [==============================] - 10s 19ms/step - loss: 1.1309 - accuracy: 0.6287
        Epoch 7/10
        497/497 [==============================] - 10s 19ms/step - loss: 0.9219 - accuracy: 0.6977
        Epoch 8/10
        497/497 [==============================] - 10s 19ms/step - loss: 0.7877 - accuracy: 0.7434
        Epoch 9/10
        497/497 [==============================] - 10s 19ms/step - loss: 0.6535 - accuracy: 0.7819
        Epoch 10/10
        497/497 [==============================] - 10s 19ms/step - loss: 0.5801 - accuracy: 0.8146
        331/331 - 2s - loss: 0.2298 - accuracy: 0.9321
```
The results this time were substantially better.

3) I decided to play around a bit and doubled the number of filters in the convolutional layer
```
        Epoch 1/10
        497/497 [==============================] - 20s 38ms/step - loss: 2.7240 - accuracy: 0.3929
        Epoch 2/10
        497/497 [==============================] - 19s 39ms/step - loss: 1.0392 - accuracy: 0.7009
        Epoch 3/10
        497/497 [==============================] - 19s 39ms/step - loss: 0.6727 - accuracy: 0.8039
        Epoch 4/10
        497/497 [==============================] - 18s 37ms/step - loss: 0.5106 - accuracy: 0.8510
        Epoch 5/10
        497/497 [==============================] - 19s 37ms/step - loss: 0.4257 - accuracy: 0.8759
        Epoch 6/10
        497/497 [==============================] - 18s 37ms/step - loss: 0.3672 - accuracy: 0.8915
        Epoch 7/10
        497/497 [==============================] - 19s 38ms/step - loss: 0.3337 - accuracy: 0.9041
        Epoch 8/10
        497/497 [==============================] - 19s 37ms/step - loss: 0.3275 - accuracy: 0.9064
        Epoch 9/10
        497/497 [==============================] - 18s 37ms/step - loss: 0.2986 - accuracy: 0.9145
        Epoch 10/10
        497/497 [==============================] - 18s 37ms/step - loss: 0.2989 - accuracy: 0.9161
        331/331 - 4s - loss: 0.1550 - accuracy: 0.9599
```
The accuracy seemed to have increased a bit but it took longer than before to compute.

4) I brought down the number of filters in the convolutional layers back to 32 and added 2 additional hidden layers both with 128 units
```
        Epoch 1/10
        497/497 [==============================] - 13s 24ms/step - loss: 2.4735 - accuracy: 0.3825
        Epoch 2/10
        497/497 [==============================] - 10s 20ms/step - loss: 0.8648 - accuracy: 0.7494
        Epoch 3/10
        497/497 [==============================] - 13s 26ms/step - loss: 0.4582 - accuracy: 0.8692
        Epoch 4/10
        497/497 [==============================] - 13s 25ms/step - loss: 0.3301 - accuracy: 0.9109
        Epoch 5/10
        497/497 [==============================] - 13s 26ms/step - loss: 0.2452 - accuracy: 0.9309
        Epoch 6/10
        497/497 [==============================] - 14s 28ms/step - loss: 0.2256 - accuracy: 0.9404
        Epoch 7/10
        497/497 [==============================] - 13s 25ms/step - loss: 0.1628 - accuracy: 0.9554
        Epoch 8/10
        497/497 [==============================] - 12s 24ms/step - loss: 0.1496 - accuracy: 0.9614
        Epoch 9/10
        497/497 [==============================] - 13s 25ms/step - loss: 0.1524 - accuracy: 0.9608
        Epoch 10/10
        497/497 [==============================] - 9s 19ms/step - loss: 0.1110 - accuracy: 0.9701
        331/331 - 1s - loss: 0.1665 - accuracy: 0.9645
```
The results were slightly better than last time

5) I added 2 more hidden layers just to see what change it would make to the final result
```
        Epoch 1/10
        497/497 [==============================] - 14s 26ms/step - loss: 2.6237 - accuracy: 0.3136
        Epoch 2/10
        497/497 [==============================] - 9s 18ms/step - loss: 1.2789 - accuracy: 0.5939
        Epoch 3/10
        497/497 [==============================] - 15s 30ms/step - loss: 0.7573 - accuracy: 0.7610
        Epoch 4/10
        497/497 [==============================] - 14s 28ms/step - loss: 0.4811 - accuracy: 0.8563
        Epoch 5/10
        497/497 [==============================] - 13s 26ms/step - loss: 0.3398 - accuracy: 0.9044
        Epoch 6/10
        497/497 [==============================] - 11s 23ms/step - loss: 0.2920 - accuracy: 0.9215
        Epoch 7/10
        497/497 [==============================] - 13s 27ms/step - loss: 0.2968 - accuracy: 0.9220
        Epoch 8/10
        497/497 [==============================] - 10s 20ms/step - loss: 0.2083 - accuracy: 0.9465
        Epoch 9/10
        497/497 [==============================] - 12s 23ms/step - loss: 0.1837 - accuracy: 0.9554
        Epoch 10/10
        497/497 [==============================] - 13s 25ms/step - loss: 0.1678 - accuracy: 0.9560
        331/331 - 3s - loss: 0.1619 - accuracy: 0.9625
```
This dosen't seem to have affected the accuracy or loss much as they are somewhat similar to the results obtained last time

5) I brought back the number of hidden layers back to 2 and doubled the number of units in each. I also increased the dropout to 0.7 so as to avoid overfitting.
```
        Epoch 1/10
        497/497 [==============================] - 17s 33ms/step - loss: 2.8528 - accuracy: 0.3114
        Epoch 2/10
        497/497 [==============================] - 11s 22ms/step - loss: 1.2252 - accuracy: 0.6332
        Epoch 3/10
        497/497 [==============================] - 9s 17ms/step - loss: 0.6526 - accuracy: 0.8067
        Epoch 4/10
        497/497 [==============================] - 9s 17ms/step - loss: 0.4315 - accuracy: 0.8750
        Epoch 5/10
        497/497 [==============================] - 9s 18ms/step - loss: 0.2971 - accuracy: 0.9173
        Epoch 6/10
        497/497 [==============================] - 9s 17ms/step - loss: 0.2466 - accuracy: 0.9330
        Epoch 7/10
        497/497 [==============================] - 9s 18ms/step - loss: 0.2212 - accuracy: 0.9413
        Epoch 8/10
        497/497 [==============================] - 9s 17ms/step - loss: 0.1887 - accuracy: 0.9546
        Epoch 9/10
        497/497 [==============================] - 9s 18ms/step - loss: 0.1694 - accuracy: 0.9572
        Epoch 10/10
        497/497 [==============================] - 9s 18ms/step - loss: 0.1519 - accuracy: 0.9649
        331/331 - 2s - loss: 0.4119 - accuracy: 0.9224
```
But that seems to have brought down the accuracy and increased the loss.

6) This time, I decided to remove the dropout altogether, ignoring the possibility of overfitting, just for the sake of experimentation. The number of units in the hidden layer were also brought back to 128
```
        Epoch 1/10
        497/497 [==============================] - 10s 19ms/step - loss: 1.7365 - accuracy: 0.6075
        Epoch 2/10
        497/497 [==============================] - 11s 22ms/step - loss: 0.4285 - accuracy: 0.8843
        Epoch 3/10
        497/497 [==============================] - 10s 21ms/step - loss: 0.2586 - accuracy: 0.9307
        Epoch 4/10
        497/497 [==============================] - 10s 20ms/step - loss: 0.1933 - accuracy: 0.9461
        Epoch 5/10
        497/497 [==============================] - 12s 23ms/step - loss: 0.1770 - accuracy: 0.9535
        Epoch 6/10
        497/497 [==============================] - 10s 20ms/step - loss: 0.0975 - accuracy: 0.9731
        Epoch 7/10
        497/497 [==============================] - 10s 20ms/step - loss: 0.1313 - accuracy: 0.9667
        Epoch 8/10
        497/497 [==============================] - 10s 20ms/step - loss: 0.1188 - accuracy: 0.9698
        Epoch 9/10
        497/497 [==============================] - 10s 20ms/step - loss: 0.1107 - accuracy: 0.9735
        Epoch 10/10
        497/497 [==============================] - 10s 20ms/step - loss: 0.1155 - accuracy: 0.9715
        331/331 - 2s - loss: 0.2138 - accuracy: 0.9591
```


## Conclusion
The best possible model seemed to be in case 4 yielding a loss of 0.1665 and an accuracy of 0.9645. Here, i used 2 convolutional-Max Pooling layer sets with 32 filters in each convolutional layer, and 3 hidden layers, each with 128 units, besides the input and output layers. Also, a dropout of 0.5 was included.
