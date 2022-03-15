# -*- coding: utf-8 -*-

#Inspired by exercise 5.2.4

from matplotlib.pylab import figure, subplot, plot, xlabel, ylabel, hist, show
import sklearn.linear_model as lm


############REGRESSION WITH ONLY FEW ATTRIBUTES:
# from Regression_preparation import *
# # Fit ordinary least squares regression model
# model = lm.LinearRegression()
# model.fit(X,y)

# # Predict alcohol content
# y_est = model.predict(X)
# residual = y_est-y

# # Display scatter plot
# figure()
# subplot(2,1,1)
# plot(y, y_est, '.')
# xlabel('Bill Length (true)'); ylabel('Bill Length (estimated)');
# subplot(2,1,2)
# hist(residual,40)

# # plt.savefig('images/regress_prep_chosen_attributes.pdf',bbox_inches = 'tight')
# show()

############## REGRESSION WITH ONE-HOT
from Regress_prep_loo import *
model = lm.LinearRegression()
model.fit(X,y)

y_est = model.predict(X)
residual = y_est-y

# Display scatter plot
# figure()
# subplot(2,1,1)
# plot(y, y_est, '.')
# xlabel('Bill Length (true)'); ylabel('Bill Length (estimated)');
# subplot(2,1,2)
# hist(residual,40)

# # plt.savefig('images/regress_prep_loo.pdf',bbox_inches = 'tight')
# show()

###plot scatter with equal axis 
figure()
plot(y, y_est, '.')
xlabel('Bill Length (true)'); ylabel('Bill Length (estimated)');
plt.axis('square')
plt.xlim(35, 55)
plt.ylim(35, 55)


# plt.savefig('images/regress_prep_loo_equal_axis.pdf',bbox_inches = 'tight')
show()


