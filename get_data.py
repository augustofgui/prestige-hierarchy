import pandas as pd
import numpy as np
import seaborn as sns
import networkx as nx
import os
os.environ['MPLCONFIGDIR'] = "/scisci/prestige-hierarchy"
from matplotlib import pyplot as plt
import pyalex 

def chunks(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

def getInternationalInstitutionsIds(ids):
    international_ids = []
    ids = chunks(ids, 100)
    for chunk in ids:
        id = '|'.join(chunk)
        query = pyalex.Institutions() \
            .filter(openalex=id).paginate(per_page=100)
        for page in query:
            for json in page:
                if json["country_code"] != "BR":
                    international_ids.append(json["id"].split("/")[3])

    return international_ids

pyalex.config.email = ""

raw_df = pd.read_csv('data/authors.csv')
df = raw_df.dropna(subset=['author_id', 'institution_id']).reset_index(drop=True)

gp_df = df[['gp_code', 'gp_name', 'gp_score', 'institution_acr', 'institution_id']].reset_index(drop=True)
gp_professors_count_df = gp_df.groupby('gp_code').size().reset_index(name='professors_count')

gp_data_df = pd.read_csv('data/br-capes-colsucup-prog-2022-2023-11-30.csv', sep=";")

columns = {
    'CD_PROGRAMA_IES': 'gp_code',
    'AN_INICIO_CURSO': 'gp_start_year'
}
csbr_gp_df = gp_data_df[gp_data_df['CD_PROGRAMA_IES'].isin(gp_df['gp_code'])].reset_index(drop=True)
csbr_gp_df = csbr_gp_df[columns.keys()].rename(columns=columns)
csbr_gp_df = csbr_gp_df
csbr_gp_df['gp_start_year'] = csbr_gp_df['gp_start_year'].str.split('/', expand=True)[0].astype('int32')
gp_df = gp_df.merge(csbr_gp_df)
gp_df = gp_df.drop_duplicates().reset_index(drop=True)

gp_df = gp_df.merge(gp_professors_count_df)
gp_df['gp_weighted_score'] = gp_df['gp_score'] * gp_df['professors_count'] 
institution_new_score_df = gp_df.groupby('institution_id').agg(
    total_score=('gp_weighted_score', 'sum'),
    total_professors_count=('professors_count', 'sum'),
    num_programs=('gp_code', 'count'),
    gp_start_year=('gp_start_year', 'min')
).reset_index()
institution_new_score_df['institution_age'] = 2022 - institution_new_score_df['gp_start_year']
institution_new_score_df['weighted_score'] = institution_new_score_df['total_score'] / institution_new_score_df['total_professors_count']
institution_new_score_df = institution_new_score_df.drop('total_score', axis=1).sort_values(by='weighted_score', ascending=False)
institution_df = df[['institution_acr', 'institution_id']].drop_duplicates(subset=['institution_id']).reset_index(drop=True).merge(institution_new_score_df)
print(institution_df.head())

institution_edges = df[['phd_institution_id', 'institution_id']].rename(columns={'institution_id': 'target', 'phd_institution_id': 'source'}).groupby(['source', 'target']).size().reset_index(name='weight')
institution_edges_br_only = institution_edges[institution_edges['source'].isin(institution_df['institution_id'])]
institution_edges_international = institution_edges[~institution_edges['source'].isin(institution_df['institution_id'])]
print(institution_edges.info())
print(institution_edges_br_only.info())
print(institution_edges_international.info())

# international_ids = getInternationalInstitutionsIds(institution_edges_international["source"])

# institution_edges_international = institution_edges_international[institution_edges_international["source"].isin(international_ids)]
# international_df = institution_edges_international.rename(columns={'target': 'institution_id'}).groupby(['institution_id'])['weight'].sum().reset_index(name='international_hires')
# print(international_df.info())
# print(international_df.head())

number = df[['phd_institution_id', 'institution_id']].rename(columns={'institution_id': 'target', 'phd_institution_id': 'source'})
number = number[number['source'].isin(institution_df['institution_id'])]
number.info()

self_hires = institution_edges_br_only[institution_edges_br_only['source'] == institution_edges_br_only['target']]
self_hires_edges = self_hires.groupby('target').size().reset_index(name='self_hires').rename(columns={"target": "institution_id"})
G = nx.DiGraph()

for index, row in institution_edges_br_only.iterrows():
    G.add_edge(row['source'], row['target'], weight=row['weight'])

print(G.number_of_nodes())
print(G.number_of_edges())

nx.write_graphml(G, "data/network.graphml")
nx.write_adjlist(G, "data/network.dat")

in_degree = [[node, val] for (node, val) in G.in_degree()]
out_degree = [[node, val] for (node, val) in G.out_degree()]
in_strength = [[node, val] for (node, val) in G.in_degree(weight='weight')]
out_strength = [[node, val] for (node, val) in G.out_degree(weight='weight')]

degree_df = pd.DataFrame(in_degree, columns=['institution_id', 'in_degree'])
print(degree_df.head())
degree_df = degree_df.merge(pd.DataFrame(out_degree, columns=['institution_id', 'out_degree']))
degree_df = degree_df.merge(pd.DataFrame(in_strength, columns=['institution_id', 'in_strength']))
degree_df = degree_df.merge(pd.DataFrame(out_strength, columns=['institution_id', 'out_strength']))

institution_df = institution_df.merge(degree_df)
# pagerank_df = pd.DataFrame.from_dict(nx.pagerank(G), orient='index', columns=['page_rank']).reset_index().rename(columns={'index': 'institution_id'})
# institution_df = institution_df.merge(pagerank_df)
new_spring_rank_df = pd.read_csv('data/new_ranking_df.csv')
institution_df = institution_df.merge(new_spring_rank_df)
institution_df = institution_df.merge(self_hires_edges, how="left").fillna(0)
institution_df = institution_df.merge(international_df, how="left").fillna(0)
institution_df.info()

plt.figure(figsize=(10, 8))
pos = nx.spring_layout(G)
nx.draw(G, pos, with_labels=True, node_color='skyblue', node_size=1500, 
        edge_color='gray', linewidths=1, font_size=15)
plt.show()
plt.savefig("figures/plot.pdf")

np.savetxt(r'data/BRCS_adjacency.dat', institution_edges_br_only.values, fmt='%s %s %d')

# spring_rank_df = pd.read_table("data/BRCS_SpringRank_a0.0_l0_1.0_l1_1.0.dat", sep="\s+", names=['institution_id', 'spring_rank'])
# institution_df = institution_df.merge(spring_rank_df)

# citescore_df = pd.read_csv('data/works_by_institution.csv')
# institution_df = institution_df.merge(citescore_df)

stats_df = pd.read_csv("data/faculty_works_data.csv")
stats_df = stats_df.drop(columns=['cited_by_counts'])[['institution_id', 'faculty_production', 'h_index', 'i10_index']]
institution_df = institution_df.merge(stats_df)

institution_df = institution_df.sort_values('new_spring_rank', ascending=False).reset_index(drop=True)
institution_df.to_csv("institution_df.csv", index=False)


# institution_normalized_df = institution_df[['total_professors_count', 'num_programs', 'weighted_score', 'spring_rank', 'faculty_production', 'h_index', 'i10_index']].div(institution_df['institution_age'], axis=0)
institution_normalized_df = institution_df[['total_professors_count', 'num_programs', 'weighted_score', 'new_spring_rank', 'new_shifted_spring_rank', 'faculty_production', 'h_index', 'i10_index', 'in_degree', 'out_degree', 'in_strength', 'out_strength']].div(institution_df['num_programs'], axis=0)

# institution_normalized_df['spring_rank'] = institution_normalized_df[['spring_rank']].div(institution_df['institution_age'], axis=0)

institution_normalized_df.to_csv("institution_normalized_df.csv", index=False)
