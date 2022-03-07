import numpy as np
import pandas as pd
import networkx as nx
import json

## Utilities
def read_Z(data_file='data/adjacency_matrix_31-12-2019.csv', t=10):
    """
    Build the Z matrix from the use table.
    
    * Z[i, j] = sales from sector i to sector j
    
    """
    
    df1 = pd.read_csv(data_file)
    df1 = df1.set_index("country")

    df2 = df1.replace(np.nan, 0)          # replace nan with 0

    df3 = df2.replace("...", 0)          # replace ... with 0

    countries = list(df3.index)
    countries = np.array(countries)
    countries = np.where(countries == 'CH', 'SW', countries)

    Z = np.asarray(df3.values.tolist(), dtype=np.float64)
    Z_visual = np.where(Z < t, 0, Z)
    
    output = {'Z':Z,'Z_visual':Z_visual, 'countries':countries}
    return output


## Chapter data

def introduction():
    """
    Load data used in Introduction chapter. 

        Returns:
            ch_data (dict): Dictionary of data names and associated data objects. Note: some data objects are further nested as dictionaries. 
    """
    
    ch_data = {}

    ## Crude oil
    fl = "data/crude_oil_sitcr2_3330_yr2019/data.csv"
    crude_oil = pd.read_csv(fl, dtype={'product_id': str})

    exporters = crude_oil.groupby(by=["location_code"]).sum().sort_values("export_value", ascending=False)[:10].index

    importers = crude_oil.groupby(by=["partner_code"]).sum().sort_values("export_value", ascending=False)[:21].index
    importers = set(importers.drop("ANS"))

    # Aggregate Data for Rest of the World
    row_concord = {}
    other_importers = set(crude_oil.partner_code.unique()).difference(importers)
    for cntry in other_importers:
        row_concord[cntry] = "ROW"
    importers.add("ROW")
    importers = pd.Index(importers, name='partner_code')

    # Aggregate Partner Locations
    crude_oil.partner_code = crude_oil.partner_code.replace(to_replace=row_concord)

    chart_data = crude_oil.groupby(by=["location_code", "partner_code"]).sum().reset_index()

    # country data
    cdata = pd.read_csv("data/crude_oil_sitcr2_3330_yr2019/regions-iso3c.csv")
    country_names = cdata[["alpha-3","name"]].set_index("alpha-3").to_dict()['name']
    country_names["ROW"] = "Rest of World"
    country_names['TWN'] = "Taiwan"
    country_names['GBR'] = "United Kingdom"

    DG_crude = nx.DiGraph()
    for idx,row in chart_data.iterrows():       
        if row.location_code not in exporters:
            continue
        if row.partner_code not in importers:
            continue
        DG_crude.add_weighted_edges_from([(country_names[row.location_code], country_names[row.partner_code], row.export_value)])

    ch_data["crude_oil"] = DG_crude

    ## aircraft_network_2019
    DATA_DIR = "data/commercial-aircraft-sitcr2-7924-yr2019"
    
    ch_data["aircraft_network_2019"] = nx.read_gexf(f"{DATA_DIR}/sitcr2-7924-aircraft-network-2019.gexf")

    f = open(f"{DATA_DIR}/sitcr2-7924-aircraft-network-2019-layout.json", "r")
    data = json.loads(f.read())
    pos = {}
    for nd in data['nodes']:
        pos[nd['id']] = np.array([nd['x'], nd['y']])
    ch_data["aircraft_network_2019_pos"] = pos
    
    ## forbes-global2000
    dfff = pd.read_csv('data/csv_files/forbes-global2000.csv')
    dfff = dfff[['Country', 'Sales', 'Profits', 'Assets', 'Market Value']]
    dfff = dfff.sort_values('Market Value', ascending=False)
    ch_data["forbes_global_2000"] = dfff
    
    ## adjacency_matrix_2019
    ch_data["adjacency_matrix_2019"] = read_Z(data_file='data/csv_files/adjacency_matrix_31-12-2019.csv', t=0)

    return ch_data


def production():
    """
    Load data used in Production chapter. 

        Returns:
            ch_data (dict): Dictionary of data names and associated data objects. Note: some data objects are further nested as dictionaries. 
    """
    ch_data = {}

    return ch_data 


def optimal_flows():
    """
    Load data used in Optimal Flows chapter. 

        Returns:
            ch_data (dict): Dictionary of data names and associated data objects. Note: some data objects are further nested as dictionaries. 
    """
    ch_data = {}

    return ch_data


def markov_chains_and_networks():
    """
    Load data used in Markov Chains and Networks chapter. 

        Returns:
            ch_data (dict): Dictionary of data names and associated data objects. Note: some data objects are further nested as dictionaries. 
    """
    ch_data = {}

    return ch_data


def nonlinear_interactions():
    """
    Load data used in Nonlinear Interactions chapter. 

        Returns:
            ch_data (dict): Dictionary of data names and associated data objects. Note: some data objects are further nested as dictionaries. 
    """
    ch_data = {}

    return ch_data


def appendix():
    """
    Load data used in Appendix chapter. 

        Returns:
            ch_data (dict): Dictionary of data names and associated data objects. Note: some data objects are further nested as dictionaries. 
    """
    ch_data = {}

    return ch_data