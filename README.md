# Airfoil_Self_Noise
This code uses nueral network and scikit-learn to predict the aerodynamic noise generated by naca0012 airfoil. The NASA data set, obtained from a series of aerodynamic and acoustic tests of two and three-dimensional airfoil blade sections conducted in an anechoic wind tunnel (T.F. Brooks, D.S. Pope, and A.M. Marcolini. Airfoil self-noise and prediction.Technical report, NASA RP-1218, July 1989.) were used to train and validate the model.
The NASA data set comprises different size NACA 0012 airfoils at various wind tunnel speeds and angles of attack. The span of the airfoil and the observer position were the same in all of the experiments.


Attribute Information:



This problem has the following inputs: 
1. Frequency, in Hertzs. 
2. Angle of attack, in degrees. 
3. Chord length, in meters. 
4. Free-stream velocity, in meters per second. 
5. Suction side displacement thickness, in meters. 

The only output is: 

6. Scaled sound pressure level, in decibels. 

Data are available at https://archive.ics.uci.edu/ml/datasets/airfoil+self-noise

 
