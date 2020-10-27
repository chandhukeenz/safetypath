# -*- coding: utf-8 -*-
"""
Created on Sun Mar 15 17:37:15 2020

@author: chandhana
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import KMeans
from gmplot import gmplot


gmap = gmplot.GoogleMapPlotter(11.0168, 76.9558, 15) #coimbatore coordinates and resolution
#dataset = pd.read_csv("SPFINAL EXCEL.xlsx",nrows=240)
dataset=pd.read_excel('SPFINAL EXCEL.xlsx',nrows=240)
X = dataset.iloc[:,[0,1]].values

#
#wcss=[]
##%matplotlib notebook
#for i in range(1,11):
#    kmeans= KMeans(n_clusters=i,init='k-means++',max_iter=300,n_init=10,random_state=0)
#    kmeans.fit(X)
#    wcss.append(kmeans.inertia_)          #sum of squares
#plt.plot(range(1,11),wcss)
#plt.title('the elbow method')
#plt.xlabel('number of clusters')
#plt.ylabel('wcss')
#plt.show()


nc=20

kmeans = KMeans(n_clusters= nc,init='k-means++',max_iter=300,n_init=10,random_state=0)
y_kmeans = kmeans.fit_predict(X)


#print(y_kmeans)

#print(kmeans.cluster_centers_[:,:])


G=dataset.iloc[:,[8]].values          #taking danger index


h=np.zeros(nc)                   #for storing total danger index of clusters
m=np.zeros(nc)                   #no of places in each cluster

x3=np.zeros(239)
for i in range(0,239,1):
    j=y_kmeans[i]
    h[j]=h[j]+np.array(G[i,0])
    x3[i]=np.array(G[i,0])
    m[j]=m[j]+1
danger_index=[]                  #for storing danger index of clusters
for i in range(0,nc,1):
    danger_index.append(h[i]/m[i])
    
    
#%matplotlib notebook
h1=plt.hist(x3,bins=4)
#h1[1]

color=[]
tag=[]
for i in range (0,nc,1):
    if(danger_index[i]>=h1[1][0] and danger_index[i]<h1[1][1]):
        color.append('white')
        tag.append('highly safe')
    elif(danger_index[i]>=h1[1][1] and danger_index[i]<h1[1][2]):
        color.append('blue')
        tag.append('safe')
    elif(danger_index[i]>=h1[1][2] and danger_index[i]<h1[1][3]):
        color.append('yellow')
        tag.append('careful')
    elif(danger_index[i]>=h1[1][3] and danger_index[i]<=h1[1][4]):
        color.append('red')
        tag.append('crime prone')


#%matplotlib notebook
for i in range(0,nc,1):
    gmap.scatter(X[y_kmeans==i,0],X[y_kmeans==i,1],s=5,c=color[i],label=tag[i],marker= False)
gmap.scatter(kmeans.cluster_centers_[:,0],kmeans.cluster_centers_[:,1],s=20,c='magenta',label='central area',marker=False)
gmap.draw("chandhu.html")
clustlat=kmeans.cluster_centers_[:,0]
clustlong=kmeans.cluster_centers_[:,1]

startlat=input("starting latitude")
startlong=input("starting longitude")
stoplat=input("stopping latitude")
stoplong=input("stopping latitude")
#p=(startlat,startlong:stoplat,stoplong)

# Import the library to make the request to the TomTom API
import requests
# Make the Request (Dont foreget to change the API key)
#r= requests.get("https://api.tomtom.com/routing/1/calculateRoute/11.02268,76.942616:11.01474,76.954777/xml?key=<yourkey>")
r= requests.get("https://api.tomtom.com/routing/1/calculateRoute/%s,%s:%s,%s/xml?key=GpStJHpdsevBpRiaQU5wYzLeIQuCCEcI"%(startlat,startlong,stoplat,stoplong))
# Print out the response to make sure it went through
print(r)


# Import the Beautiful Soup Library
from bs4 import BeautifulSoup
# Grab the content from our request
c = r.content
# Turn the XML data into a human readable format
soup = BeautifulSoup(c)
# Print out the information
print(soup.prettify())


# Find all the tags that contain a point in our route
Points = soup.find_all('point')
# Initialize our 2 arrays that will contain all the points
lat = []
long = []
# Iterate through all the points and add the lat and long to the correct array
for point in Points:
    lat.append(point['latitude'])
    long.append(point['longitude'])

rlat=np.asarray(lat, dtype='float64')
rlong=np.asarray(long, dtype='float64')

route_danger=0
count=0
for i in range(0,len(rlat),1):
    for j in range(0,20,1):
        if(np.round(rlat[i],3) == np.round(clustlat[j],3)):
            count=count+1
            route_danger=route_danger+danger_index[j]
route_danger=route_danger/count

    
rcolor=[]
rtag=[]
for i in range (0,1,1):
    if(route_danger>=0 and route_danger<2.857375):
        rcolor.append('white')
        rtag.append('highly safe')
    elif(route_danger>=2.857375 and route_danger<5.23825):
        rcolor.append('blue')
        rtag.append('safe')
    elif(route_danger>=5.23825 and route_danger<7.619125):
        rcolor.append('yellow')
        rtag.append('careful')
    elif(route_danger>=7.619125 and route_danger<=10):
        rcolor.append('red')
        rtag.append('crime prone')    
    
    
# Convert the points to floats   
#%matplotlib notebook
lat = [float(x) for x in lat]
long = [float(x) for x in long]
# Import the plotting library
#import matplotlib.pyplot as plt
gmap.scatter(lat,long,s=20,c=rcolor[0],label=rtag,marker=False)
gmap.draw("output.html")
#plt.title('Route from TomTom API')
#plt.show()   
    
