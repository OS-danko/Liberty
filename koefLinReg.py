import math

import matplotlib.pyplot as plt
from kneed import KneeLocator
from sklearn import linear_model
import numpy as np
import pandas as pd
import openpyxl
from sklearn.metrics import mean_squared_error, r2_score

pd.options.mode.chained_assignment = None
xy = pd.read_excel('1.xlsx', sheet_name='1', usecols=['x', 'y'])
kneedle1 = KneeLocator(xy.x, xy.y, S=1.0, curve='convex', direction='increasing', online=True)
list_kane1 = list(kneedle1.all_knees)
list_kane1.sort()
# print(list_kane1)
out1 = [(list_kane1[i], list_kane1[i + 1]) for i in range(0, len(list_kane1) - 1)]
mx1 = [(rec[1] - rec[0]) for rec in out1]
# print(out1)
# print(out1[np.argmax(mx)])
Ndiapz1 = np.argmax(mx1)
regr1 = linear_model.LinearRegression()
subset1 = xy[(xy.x >= out1[Ndiapz1][0]) & (xy.x <= out1[Ndiapz1][1])]
X1 = subset1.x
Y1 = subset1.y
X1_r = np.array(X1).reshape((len(X1), 1))
Y1_r = np.array(Y1).reshape((len(Y1), 1))
regr1.fit(X1_r, Y1_r)
y_pred1 = regr1.predict(X1_r)
# The coefficients
a1 = regr1.coef_
b1 = regr1.intercept_
print('Coefficients convex+increasing: \n', 'a=', a1, '\tb=', b1)
# The mean squared error
mse1 = mean_squared_error(Y1_r, y_pred1)
print('Mean squared error: %.2f' % mse1)
# The coefficient of determination: 1 is perfect prediction
r2_1 = r2_score(Y1_r, y_pred1)
print('Coefficient of determination: %.2f' % r2_1)
i=0
while abs(Y1_r[i]-(X1_r[i]*a1[0]+b1[0]))>=1:
    i+=1
point_left1=X1_r[i]
i=len(Y1_r)-1
while abs(Y1_r[i]-(X1_r[i]*a1[0]+b1[0]))>=1:
    i-=1
point_right1=X1_r[i]
print(point_left1,'\t',point_right1)
print()

kneedle2 = KneeLocator(xy.x, xy.y, S=1.0, curve='concave', direction='increasing', online=True)
list_kane2 = list(kneedle2.all_knees)
list_kane2.sort()
# print(list_kane1)
out2 = [(list_kane2[i], list_kane2[i + 1]) for i in range(0, len(list_kane2) - 1)]
mx2 = [(rec[1] - rec[0]) for rec in out2]
# print(out1)
# print(out1[np.argmax(mx)])
Ndiapz2 = np.argmax(mx2)
regr2 = linear_model.LinearRegression()
subset2 = xy[(xy.x >= out2[Ndiapz2][0]) & (xy.x <= out2[Ndiapz2][1])]
X2 = subset2.x
Y2 = subset2.y
X2_r = np.array(X2).reshape((len(X2), 1))
Y2_r = np.array(Y2).reshape((len(Y2), 1))
regr2.fit(X2_r, Y2_r)
y_pred2 = regr2.predict(X2_r)
# The coefficients
a2 = regr2.coef_
b2 = regr2.intercept_
print('Coefficients concave+increasing: \n', 'a=', a2, '\tb=', b2)
# The mean squared error
mse2 = mean_squared_error(Y2_r, y_pred2)
print('Mean squared error: %.2f' % mse2)
# The coefficient of determination: 1 is perfect prediction
r2_2 = r2_score(Y2_r, y_pred2)
print('Coefficient of determination: %.2f' % r2_2)
i=0
while abs(Y2_r[i]-(X2_r[i]*a2[0]+b2[0]))>=1:
    i+=1
point_left2=X2_r[i]
i=len(Y2_r)-1
while abs(Y2_r[i]-(X2_r[i]*a2[0]+b2[0]))>=1:
    i-=1
point_right2=X2_r[i]
print(point_left2,'\t',point_right2)
print()



if r2_1 >= r2_2 or mse1 <= mse2:
    X = X1
    y_pred = y_pred1
    out = out1
    r2 = r2_1
    mse = mse1
    Ndiapz = Ndiapz1
    ttl = 'convex+increasing'
    a=a1
    b=b1
    point_left =point_left1
    point_right=point_right1
else:
    X = X2
    y_pred = y_pred2
    out = out2
    r2 = r2_2
    mse = mse2
    Ndiapz = Ndiapz2
    ttl = 'concave+increasing'
    a=a2
    b=b2
    point_left = point_left2
    point_right = point_right2

subset = xy[(xy.x >= point_left[0]) & (xy.x <= point_right[0])]
subset['yp']=subset['x']*a[0]+b[0]
subset['(yp-y)^2']=(subset['yp']-subset['y'])**2
mse=(subset['(yp-y)^2'].sum()/subset['(yp-y)^2'].count())**0.5
#print(subset.head(10))
print('α=',math.degrees(math.atan(a[0])))
xy.plot(kind='line', x='x', y='y', color='red')
plt.plot(X, y_pred, color='blue', linewidth=2)
plt.vlines([point_left, point_right], plt.ylim()[0], plt.ylim()[1], linestyles='dashed')
plt.title(ttl)
plt.xlabel('Девормативность, ε=ΔL/H')
plt.ylabel('Нагрузка, σ=P/S, МПа')
plt.legend(['Исходные данные', 'Линейная регрессия'])
plt.annotate(text='a=' + str(round(point_left[0],4)), xy=(point_left, plt.ylim()[0]), fontsize=10)
plt.annotate(text='b=' + str(round(point_right[0],4)), xy=(point_right, plt.ylim()[0]), fontsize=10)
plt.annotate(text='R^2=' + str(round(r2, 4)) + ' MSE=' + str(round(mse,4 ))+' α='+str(round(math.degrees(math.atan(a[0])),4))+'$^o$',
             xy=(plt.xlim()[1] * 1 / 4, plt.ylim()[1] / 3), fontsize=10)
plt.annotate(text='Linear regression equation', xy=(plt.xlim()[1] * 1 / 4, plt.ylim()[1] / 4), fontsize=10)
znak='+'
if b[0]<0:
    znak=''
plt.annotate(text='y='+str(a.round(4))+'*x'+znak+str(round(b[0],4)), xy=(plt.xlim()[1] * 1 / 4, plt.ylim()[1] / 5), fontsize=10)
plt.show()
