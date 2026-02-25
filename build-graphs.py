import pandas as pd
import networkx as nx
import os
import json

if __name__ == "__main__":
    print("CONSTRUINDO GRAFOS")
    df = pd.read_csv("processed/br-capes-colsucup-docente-deduped-institutions.csv", low_memory=False)
    print(f"    TAMANHO: {len(df)} LINHAS")

    print(f"    INSTITUIÇÕES DE GRAU: {len(df[df['degree_institution_abbr'].notna()])}")    

    df = df[df["degree_institution_abbr"].notna() & df["institution_abbr"].notna()]
    print(f"    ARESTAS VÁLIDAS: {len(df)} LINHAS")
    
    before_invalid_removal = len(df)
    df = df[
        (df["degree_institution_abbr"].astype(str).str.lower() != "invalid") &
        (df["institution_abbr"].astype(str).str.lower() != "invalid")
    ]
    if before_invalid_removal > len(df):
        print(f"    REMOVIDAS {before_invalid_removal - len(df)} LINHAS COM NÓS INVÁLIDOS")
        print(f"    ARESTAS RESTANTES: {len(df)} LINHAS")
    
    country_list = df[df["is_international"] == True]["degree_institution_abbr"].dropna().unique().tolist()
    print(f"    INSTITUIÇÕES INTERNACIONAIS: {len(country_list)}")
   
    br_institutions_df = pd.read_csv("processed/manual/br_institutions.csv")
    br_institutions_state_dict = dict(zip(br_institutions_df["abbr"], br_institutions_df["state"]))
    br_institutions_region_dict = dict(zip(br_institutions_df["abbr"], br_institutions_df["region"]))
    
    edges_by_year = {}
    professors_by_year = []
    for year, group in df.groupby("base_year"):
        edges = group.groupby(["degree_institution_abbr", "institution_abbr", "field_id", "field_name", "big_field_id", "big_field_name"]).size().reset_index(name="weight")
        edges_by_year[int(year)] = edges
        n_prof_global = group["professor_id"].nunique()
        professors_by_year.append({"year": f"{year}", "type": "global", "n_prof": int(n_prof_global), "big_field_id": None, "big_field_name": None, "field_id": None, "field_name": None})
        n_prof_by_big_field = (
            group
            .groupby(["big_field_id", "big_field_name"])["professor_id"]
            .nunique()
            .reset_index(name="num_professors")
        )
        for big_field_id, big_field_name, n_prof in n_prof_by_big_field.values:
            professors_by_year.append({"year": f"{year}", "type": "big_field", "big_field_id": int(big_field_id), "big_field_name": big_field_name, "n_prof": int(n_prof), "field_id": None, "field_name": None})
        n_prof_by_field = (
            group
            .groupby(["field_id", "field_name", "big_field_id", "big_field_name"])["professor_id"]
            .nunique()
            .reset_index(name="num_professors")
        )
        for field_id, field_name, big_field_id, big_field_name, n_prof in n_prof_by_field.values:
            professors_by_year.append({"year": f"{year}", "type": "field", "field_id": int(field_id), "field_name": field_name, "big_field_id": int(big_field_id), "big_field_name": big_field_name, "n_prof": int(n_prof)})
    print(f"    ANOS: {sorted(edges_by_year.keys())}")
    
    graphs_by_year = {year: nx.from_pandas_edgelist(
        edges,
        source="degree_institution_abbr",
        target="institution_abbr",
        edge_attr=["weight", "field_id", "field_name", "big_field_id", "big_field_name"],
        create_using=nx.MultiDiGraph()
    ) for year, edges in edges_by_year.items()}
    
    for G in graphs_by_year.values():
        intl_flag = {n: (n in country_list) for n in G.nodes()}
        nx.set_node_attributes(G, intl_flag, "international")
        nx.set_node_attributes(G, br_institutions_state_dict, "state")
        nx.set_node_attributes(G, br_institutions_region_dict, "region")
        nx.write_graphml(G, f"processed/graphs/all_fields/test/{year}.graphml")
    time_windows = [(2004,2024), (2011,2020)]
    print("    CONSTRUINDO GRAFOS POR JANELAS DE TEMPO")
    time_window_field_graphs_count = 0
    time_window_big_field_graphs_count = 0
    for start_year, end_year in time_windows:
        window_df = df[(df["base_year"] >= start_year) & (df["base_year"] <= end_year)]
        max_year_per_prof = window_df.groupby("professor_id")["base_year"].transform("max")
        min_year_per_prof = window_df.groupby("professor_id")["base_year"].transform("min")
        min_df = window_df[window_df["base_year"] == min_year_per_prof]
        window_df = window_df[window_df["base_year"] == max_year_per_prof]
        edges = window_df.groupby(["degree_institution_abbr", "institution_abbr", "field_id", "field_name", "big_field_id", "big_field_name", "base_year"]).size().reset_index(name="weight")
        min_edges = min_df.groupby(["degree_institution_abbr", "institution_abbr", "field_id", "field_name", "big_field_id", "big_field_name", "base_year"]).size().reset_index(name="weight")
        min_G = nx.from_pandas_edgelist(
            min_edges,
            source="degree_institution_abbr",
            target="institution_abbr",
            edge_attr=["weight", "base_year", "field_id", "field_name", "big_field_id", "big_field_name"],
            create_using=nx.MultiDiGraph()
        )
        intl_flag = {n: (n in country_list) for n in G.nodes()}
        nx.set_node_attributes(G, intl_flag, "international")
        nx.set_node_attributes(min_G, br_institutions_state_dict, "state")
        nx.set_node_attributes(min_G, br_institutions_region_dict, "region")
        nx.write_graphml(min_G, f"processed/graphs/all_fields/test/{start_year}-{end_year}-first-appearance.graphml")
        G = nx.from_pandas_edgelist(
            edges,
            source="degree_institution_abbr",
            target="institution_abbr",
            edge_attr=["weight", "base_year", "field_id", "field_name", "big_field_id", "big_field_name"],
            create_using=nx.MultiDiGraph()
        )
        intl_flag = {n: (n in country_list) for n in G.nodes()}
        nx.set_node_attributes(G, intl_flag, "international")
        nx.set_node_attributes(G, br_institutions_state_dict, "state")
        nx.set_node_attributes(G, br_institutions_region_dict, "region")
        nx.write_graphml(G, f"processed/graphs/all_fields/test/{start_year}-{end_year}.graphml")
        
        n_prof_global = window_df["professor_id"].nunique()
        professors_by_year.append({"year": f"{start_year}-{end_year}", "type": "global", "n_prof": int(n_prof_global), "big_field_id": None, "big_field_name": None, "field_id": None, "field_name": None})
        n_prof_by_big_field = (
            window_df
            .groupby(["big_field_id", "big_field_name"])["professor_id"]
            .nunique()
            .reset_index(name="num_professors")
        )
        for big_field_id, big_field_name, n_prof in n_prof_by_big_field.values:
            professors_by_year.append({"year": f"{start_year}-{end_year}", "type": "big_field", "big_field_id": int(big_field_id), "big_field_name": big_field_name, "n_prof": int(n_prof), "field_id": None, "field_name": None})
        n_prof_by_field = (
            window_df
            .groupby(["field_id", "field_name", "big_field_id", "big_field_name"])["professor_id"]
            .nunique()
            .reset_index(name="num_professors")
        )
        for field_id, field_name, big_field_id, big_field_name, n_prof in n_prof_by_field.values:
            professors_by_year.append({"year": f"{start_year}-{end_year}", "type": "field", "field_id": int(field_id), "field_name": field_name, "big_field_id": int(big_field_id), "big_field_name": big_field_name, "n_prof": int(n_prof)})

    professors_by_year_df = pd.DataFrame(professors_by_year)
    professors_by_year_df.to_csv("processed/professors_by_year.csv", index=False)

    print("CONSTRUÇÃO DE GRAFOS CONCLUÍDA")
