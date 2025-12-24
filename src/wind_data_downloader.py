import gc
import numpy as np
import pandas as pd
import requests
import asyncio
import aiohttp
import dask.array as da
import xarray as xr
from io import StringIO
from typing import Literal

class WindDataDownloader:
    def __init__(self, city_coords=None, zarr_path=None):
        self.download_limit = 30
        self.sem = asyncio.Semaphore(self.download_limit)
        self.city_coords = city_coords or self.download_city_coords()
        self.zarr_path = zarr_path or '../data/wind_data_all'
        self.max_len = max(len(city) for city in self.city_coords.keys())

    def download_city_coords(self):
        url = "https://raw.githubusercontent.com/epogrebnyak/ru-cities/main/assets/towns.csv"
        df = pd.read_csv(url)
        city_coords = dict()
        for _, data in df[['city', 'lon', 'lat']].iterrows():
            city_coords[data['city']] = (data['lon'], data['lat'])
        print(f'City coords downloaded. {len(city_coords)} cities')
        return city_coords

    def add_heights(heights=np.array([20, 30])):
        pass
    
    async def download_wind_df_async(
        self, 
        coords: tuple, 
        start_year: int,
        end_year: int,
        session,
        heights=np.array([10, 20, 30]),
        frequency: Literal["daily", "hourly"] = 'hourly'
    ):
        """
        Скачивает данные ветра NASA POWER за несколько лет подряд.
        """

        url = f"https://power.larc.nasa.gov/api/temporal/{frequency}/point"
        params = {
            "parameters": "WS10M,WS50M",
            "community": "RE",
            "longitude": coords[0],
            "latitude": coords[1],
            "start": f"{start_year}0101",
            "end": f"{end_year}1231",
            "format": "CSV"
        }

        async with self.sem:
            async with session.get(url, params=params) as r:
                if r.status != 200:
                    print(f"Ошибка: {r.status}")
                    return None
                text = await r.text()

        df = pd.read_csv(StringIO(text), skiprows=10)
        mask = {'year': df['YEAR'], 'month': df['MO'], 'day': df['DY']}
        if frequency == "hourly":
            mask["hour"] = df["HR"]

        df.index = pd.to_datetime(mask)
        df = df[['WS10M', 'WS50M']].replace(-999, np.nan)
        df.columns = [10, 50]

        alpha = 0.1715
        original_heights = df.columns
        add_heights = np.array([h for h in heights if h not in original_heights])
        if len(add_heights) > 0:
            wind_orig = df[original_heights].values
            wind_new = (wind_orig[:, 0:1] * (add_heights/10)**alpha + 
                        wind_orig[:, 1:2] * (add_heights/50)**alpha) / 2
            df_new = pd.DataFrame(wind_new, columns=add_heights, index=df.index)
            df = pd.concat([df, df_new], axis=1)

        df.columns.name = 'height'
        df.index.name = 'time'
        df = df[sorted(heights)].round(2)
        return df

    def df_append_to_zarr(self, df, city, path, it_first):
        darr = da.from_array(df.values.T[None, :, :])
        wind_data = xr.DataArray(
            darr,
            dims=['city', 'height', 'time'],
            coords={
                'city': np.array([city], dtype=f'U{self.max_len}'),
                'height': df.columns,
                'time': df.index
            },
            name='wind_data'
        )

        if it_first:
            wind_data.to_zarr(path, encoding={'wind_data': {'chunks': (10,10,10000)}}, mode='w')
        else:
            wind_data.to_zarr(path, append_dim='city', mode='a')

    async def download_cities_to_zarr(
            self, 
            start_year, end_year, 
            heights=np.array([10, 20, 30]), 
            frequency: Literal["daily", "hourly"] = 'hourly'
        ):

        cities = list(self.city_coords.items()) # [(city, city_cords)]
        total = len(cities)
        first_write = True

        async with aiohttp.ClientSession() as session:
            for batch_start in range(0, total, self.download_limit):
                batch = cities[batch_start : batch_start + self.download_limit]
                tasks = [
                    self.download_wind_df_async(city_coords, start_year, end_year, session, heights, frequency)
                    for _, city_coords in batch
                ]
                dfs = await asyncio.gather(*tasks)
                for (city, _), df in zip(batch, dfs):
                    if df is not None:
                        self.df_append_to_zarr(df, city, self.zarr_path, it_first=first_write)
                        first_write = False

                del dfs
                gc.collect()
                print(f"Батч {batch_start//self.download_limit + 1} готов")

    def make_wind_dataarray(self, wind_sources):
        """
        wind_sources: dict
            city_name -> (df, (lon, lat))
        """
        cities = []
        lons = []
        lats = []
        dfs = []

        for city, (df, (lon, lat)) in wind_sources.items():
            cities.append(city)
            lons.append(lon)
            lats.append(lat)
            dfs.append(df)

        times = dfs[0].index
        heights = [10, 30, 50]

        # Собираем массив (city, height, time)
        data = np.stack([df[heights].values.T for df in dfs], axis=0)
        
        return xr.DataArray(
            data,
            dims=["city", "height", "time"],
            coords={
                "city": cities,
                "height": heights,
                "time": times,
                "longitude": ("city", lons),
                "latitude": ("city", lats),
            },
            name="wind_speed"
        )
    
    def download_wind_dataarray(self, city_coords, start_year, end_year, frequency="hourly"):
        """
        city_coords: dict
            city_name -> (longitude, latitude)
        """
        wind_sources = {}

        for city, (lon, lat) in city_coords.items():
            df = self.download_wind_df(lon, lat, start_year, end_year, frequency)
            if df is not None:
                print(f"{city} data downloaded")
                wind_sources[city] = (df, (lon, lat))

        
        return self.make_wind_dataarray(wind_sources)