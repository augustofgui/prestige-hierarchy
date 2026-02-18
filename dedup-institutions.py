import pandas as pd

if __name__ == "__main__":
    print("DEDUPLICATING INSTITUTIONS")
    df = pd.read_csv("processed/br-capes-colsucup-docente-professors-deduplicated.csv", sep=",", low_memory=False, encoding='utf-8')
    print(f"    SIZE: {len(df)} ROWS")
    
    unique_institutions_before = df[["institution_name", "institution_abbr"]].drop_duplicates().shape[0]
    unique_degree_institutions_before = df[["degree_institution_name", "degree_institution_abbr"]].drop_duplicates().shape[0]
    print(f"    UNIQUE INSTITUTIONS (before): {unique_institutions_before}")
    print(f"    UNIQUE DEGREE INSTITUTIONS (before): {unique_degree_institutions_before}")
    
    mapping_df = pd.read_csv("processed/manual/mapping_df.csv", sep=",", low_memory=False, encoding='utf-8')
    br_institutions_df = pd.read_csv("processed/manual/br_institutions.csv", sep=",", low_memory=False, encoding='utf-8')
    
    mapping_dict = (
        mapping_df
        .groupby(["name", "abbr"])["mapped_abbr"]
        .agg(lambda x: "/".join(sorted(set(x.dropna()))))
        .to_dict()
    )
    
    br_institutions_df_unique = br_institutions_df.drop_duplicates(subset=["abbr"], keep="first")
    br_institutions_name_dict = br_institutions_df_unique.set_index("abbr")["name"].to_dict()
    br_institutions_state_dict = br_institutions_df_unique.set_index("abbr")["state"].to_dict()
    br_institutions_region_dict = br_institutions_df_unique.set_index("abbr")["region"].to_dict()
    
    df["_inst_key"] = df["institution_name"].astype(str) + "|" + df["institution_abbr"].astype(str)
    df["_degree_key"] = df["degree_institution_name"].astype(str) + "|" + df["degree_institution_abbr"].astype(str)
    
    mapping_dict_str = {f"{k[0]}|{k[1]}": v for k, v in mapping_dict.items()}
    df["institution_abbr_mapped"] = df["_inst_key"].map(mapping_dict_str)
    df["degree_abbr_mapped"] = df["_degree_key"].map(mapping_dict_str)
    df = df.drop(columns=["_inst_key", "_degree_key"])
    
    df["institution_abbr_mapped"] = df["institution_abbr_mapped"].astype(str).apply(
        lambda x: x.split("/") if "/" in x and x != "nan" else ([x] if x != "nan" else [pd.NA])
    )
    df["degree_abbr_mapped"] = df["degree_abbr_mapped"].astype(str).apply(
        lambda x: x.split("/") if "/" in x and x != "nan" else ([x] if x != "nan" else [pd.NA])
    )
    
    before_explode = len(df)
    df = df.explode("institution_abbr_mapped", ignore_index=True)
    df = df.explode("degree_abbr_mapped", ignore_index=True)
    print(f"    EXPLODED ROWS: {before_explode} -> {len(df)} ROWS")
    
    df["institution_abbr_mapped"] = df["institution_abbr_mapped"].astype(str).str.strip()
    df["degree_abbr_mapped"] = df["degree_abbr_mapped"].astype(str).str.strip()
    df["institution_abbr_mapped"] = df["institution_abbr_mapped"].replace("nan", pd.NA)
    df["degree_abbr_mapped"] = df["degree_abbr_mapped"].replace("nan", pd.NA)
    
    before_invalid_removal = len(df)
    df = df[
        (df["institution_abbr_mapped"].astype(str).str.lower() != "invalid") &
        (df["degree_abbr_mapped"].astype(str).str.lower() != "invalid")
    ]
    if before_invalid_removal > len(df):
        print(f"    REMOVED {before_invalid_removal - len(df)} ROWS WITH INVALID MAPPED_ABBR")
    
    df["institution_canonical_name"] = df["institution_abbr_mapped"].map(br_institutions_name_dict)
    df["institution_state"] = df["institution_abbr_mapped"].map(br_institutions_state_dict)
    df["institution_region"] = df["institution_abbr_mapped"].map(br_institutions_region_dict)
    
    df["degree_institution_canonical_name"] = df["degree_abbr_mapped"].map(br_institutions_name_dict)
    df["degree_institution_state"] = df["degree_abbr_mapped"].map(br_institutions_state_dict)
    df["degree_institution_region"] = df["degree_abbr_mapped"].map(br_institutions_region_dict)
    
    is_international = df["degree_institution_canonical_name"].isna()
    df.loc[is_international, "degree_institution_canonical_name"] = df.loc[is_international, "degree_institution_country"]
    df.loc[is_international, "degree_institution_state"] = df.loc[is_international, "degree_institution_country"]
    df.loc[is_international, "degree_institution_region"] = df.loc[is_international, "degree_institution_country"]
    df.loc[is_international, "degree_abbr_mapped"] = df.loc[is_international, "degree_institution_country"]
    df["is_international"] = is_international
    
    df = df.drop(columns=["institution_name", "institution_abbr", "degree_institution_name", "degree_institution_abbr"])
    df = df.rename(columns={"institution_abbr_mapped": "institution_abbr", "degree_abbr_mapped": "degree_institution_abbr"})
    
    unique_institutions_after = df[["institution_abbr", "institution_canonical_name"]].drop_duplicates().shape[0]
    unique_degree_institutions_after = df[["degree_institution_abbr", "degree_institution_canonical_name"]].drop_duplicates().shape[0]
    
    df.to_csv("processed/br-capes-colsucup-docente-deduped-institutions.csv", sep=",", index=False, encoding='utf-8')
    print(f"\nTOTAL: {len(df)} ROWS")
    print(f"    UNIQUE INSTITUTIONS (after): {unique_institutions_after}")
    print(f"    UNIQUE DEGREE INSTITUTIONS (after): {unique_degree_institutions_after}")
    print("DEDUPLICATION COMPLETED")
