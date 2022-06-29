import pandas as pd
import geocoder
import folium

ip = geocoder.ip("202.51.247.22")
ip.latlng

data = {'Camera Name':  ['PGPR', 'Com2'],
        'Latitude': ['1.291654', '1.294108'],
        'Longitude': ['103.780445', '103.773765']
        }

df = pd.DataFrame(data)

print (df)

x = int(input("Enter Camera Number: "))
location = [df['Latitude'][x],df['Longitude'][x]]
map = folium.Map(location=location, zoom_start=10)
folium.Marker(location).add_to(map)
map