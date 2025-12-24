import numpy as np
import xarray as xr
import pandas as pd

class HourlyConsumptionDataGenerator:
    def __init__(
            self, 
            times: xr.DataArray, 
            appliance_usage_hours: pd.DataFrame, 
            period_hours_mapping: dict
    ):
        self.rng = np.random.default_rng(42) 
        self.times = times   
        self.appliance_usage_hours = appliance_usage_hours
        self.period_hours_mapping = period_hours_mapping

    def circular_gaussian_weights(self, hours, center, sigma):
        hours = np.array(hours)

        # минимальное расстояние по кругу
        dist = np.minimum(
            np.abs(hours - center),
            24 - np.abs(hours - center)
        )

        weights = np.exp(-0.5 * (dist / sigma) ** 2)
        return weights / weights.sum()

    def circular_mean(self, hours):
        angles = 2 * np.pi * np.array(hours) / 24
        mean_angle = np.angle(np.mean(np.exp(1j * angles)))
        if mean_angle < 0:
            mean_angle += 2 * np.pi
        return 24 * mean_angle / (2 * np.pi)

    def generate_random_daily_profile(self, appliance_usage_hours, period_hours_mapping):
        """
        Генерирует случайный 24-часовой профиль потребления (Вт),
        сохраняя энергию по каждому прибору и периоду.
        """
        daily_profile = np.zeros(24)

        for _, row in appliance_usage_hours.iterrows():
            power = row['Вт']

            for period, hours in period_hours_mapping.items():
                usage_hours = row[period]
                if usage_hours <= 0:
                    continue

                hours = list(hours)
                n_hours = len(hours)

                center = self.circular_mean(hours)
                sigma = max(2, n_hours / 2)

                weights = self.circular_gaussian_weights(hours, center, sigma)

                # сколько целых часов + остаток
                full_hours = int(np.floor(usage_hours))
                remainder = usage_hours - full_hours

                # случайно выбираем часы включения
                chosen_hours = self.rng.choice(hours, size=min(full_hours, n_hours), replace=False, p=weights)

                for h in chosen_hours:
                    daily_profile[h] += power

                # остаток часа (дробная часть)
                if remainder > 0:
                    h = self.rng.choice(hours, p=weights)
                    daily_profile[h] += power * remainder

        return daily_profile

    def get_hourly_consumption(self):
        n_days = len(np.unique(self.times.dt.date))

        profiles = []

        for _ in range(n_days):
            daily = self.generate_random_daily_profile(
                self.appliance_usage_hours,
                self.period_hours_mapping
            )
            profiles.append(daily)

        profiles = np.array(profiles)  # shape: (days, 24)

        hourly_consumption_random = profiles.flatten()

        hourly_consumption = xr.DataArray(
            hourly_consumption_random,
            dims="time",
            coords={"time": self.times}
        )
        return hourly_consumption