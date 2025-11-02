import geopandas
import pandas as pd


def ped_count_csv():
    df = pd.read_csv("data/count.csv")
    print(df)


def ped_count_geojson():
    df = geopandas.read_file("data/count.geojson")
    print(df)


def ped_demand_csv():
    df = pd.read_csv("data/demand.csv")
    print(df)


def ped_demand_geojson():
    df = geopandas.read_file("data/demand.geojson")
    print(df)


def main():
    ped_count_csv()
    ped_count_geojson()
    ped_demand_csv()
    ped_demand_geojson()


if __name__ == "__main__":
    main()
