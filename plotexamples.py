import matplotlib.pyplot as plt

# Sample data
year = [1950, 1960, 1970, 1980, 1990]
gdp = [200, 400, 600, 800, 1000]

plt.plot(year,gdp,color='red',marker='+',linestyle=':')
plt.title('GDP over years')
plt.ylabel('GDP')
plt.xlabel('Year')
plt.grid()
plt.show()
