import os
import glob
import re
import math
import pandas as pd
from utils.normalization import normalize_text

def is_valid(x):
    if x is None:
        return False
    
    if isinstance(x, float) and math.isnan(x):
        return False
    
    if not isinstance(x, str):
        return False
    
    invalid_patterns = [
        r"^invalid$",
        r"^[A-Za-z]$",
        r"^(.)\1{1,}$",
        r"^\d$",
        r"^\d+$",
        r"^nao informado$",
        r"^nao informada$",
        r"^instituicao nao cadastrada$",
        r"^nao consta$",
        r"^outra$",
        r"^outro$",
        r"^ni$"
    ]

    invalid_patterns_regex = "|".join(invalid_patterns)
    
    if x == "" or x == "nan":
        return False
    if re.search(invalid_patterns_regex, x):
        return False
    if re.fullmatch(r"[^a-z0-9]+", x):
        return False
    return True

capes_columns_mapping = {
    "AN_BASE": "base_year",
    "NM_ENTIDADE_ENSINO": "institution_name",
    "SG_ENTIDADE_ENSINO": "institution_abbr",
    "NM_IES_TITULACAO": "degree_institution_name",
    "NM_IES_TIT_MAX_DOCENTE": "degree_institution_name",
    "SG_IES_TITULACAO": "degree_institution_abbr",
    "SG_IES_TIT_MAX_DOCENTE": "degree_institution_abbr",
    "NM_PAIS_IES_TITULACAO": "degree_institution_country",
    "NM_PAIS_IES_TIT_MAX_DOCENTE": "degree_institution_country",
    "AN_TITULACAO_DOCENTE": "degree_year",
    "AN_TITULACAO": "degree_year",
    "NM_GRAU_TITULACAO": "degree_level",
    "DS_TITULACAO_ATUAL_DOCENTE": "degree_level",
    "CD_AREA_AVALIACAO": "field_id",
    "ID_AREA_AVALIACAO": "field_id",
    "NM_AREA_AVALIACAO": "field_name",
    "CD_PROGRAMA_IES": "program_id",
    "NM_PROGRAMA_IES": "program_name",
    "CD_CONCEITO_PROGRAMA": "program_capes_score",
    "NM_GRAU_PROGRAMA": "program_degree",
    "NM_NIVEL_PROGRAMA": "program_degree",
    "NM_MODALIDADE_PROGRAMA": "program_type"
}

def report_removed_rows(df, func, text):
    before = len(df)
    df = func(df)
    after = len(df)
    print(f"    REMOVED {before - after} {text} ROWS")
    return df

def remove_invalid_rows(df):
    df = df[df.apply(lambda row: is_valid(row["degree_institution_name"]) or is_valid(row["degree_institution_abbr"]), axis=1)]
    return df

def normalize_columns(df):
    for column in df.columns:
        if column in ["institution_name", "institution_abbr", "degree_institution_name", "degree_institution_abbr", "degree_institution_country", "field_name", "program_name"]:
            df[column] = df[column].apply(normalize_text)
    print("    NORMALIZATION COMPLETED")
    return df

def process_data_files(data_files):
    dfs = []
    for data_file in data_files:
        print("\nPROCESSING FILE: ", data_file)
        df = pd.read_csv(data_file, encoding="latin1", sep=";", low_memory=False)
        print(f"    SIZE: {len(df)} ROWS")

        # special case for 2004-2012 capes data
        if 'CD_CONCEITO_PROGRAMA' not in df.columns:
            df['CD_CONCEITO_PROGRAMA'] = "nao consta"

        df = df.rename(columns=capes_columns_mapping)[list(set(capes_columns_mapping.values()))]

        df = report_removed_rows(df, lambda df: df[df["program_type"] == "ACADÊMICO"], "NON-ACADEMIC PROGRAMS")
        df = report_removed_rows(df, lambda df: df[df["degree_level"].str.contains("DOUTOR", case=False, na=False)], "NON-DOCTORAL DEGREES")

        df["has_masters"] = df["program_degree"].str.contains("MESTRADO", case=False, na=False)
        df["has_doctors"] = df["program_degree"].str.contains("DOUTORADO", case=False, na=False)
        df["has_professional_degree"] = df["program_degree"].str.contains("PROFISSIONAL", case=False, na=False)

        df.drop(columns=["program_type", "program_degree", "degree_level"], inplace=True)

        df = normalize_columns(df)

        df = report_removed_rows(df, remove_invalid_rows, "INVALID")

        df = df.apply(lambda col: col.map(lambda x: "invalid" if not is_valid(x) else x))

        os.makedirs("processed", exist_ok=True)
        os.makedirs("processed/by_year", exist_ok=True)

        dfs.append(df)
        for year, df_year in df.groupby("base_year"):
            df_year.to_csv(f"processed/by_year/br-capes-colsucup-docente-{year}.csv", index=False, encoding="utf-8", sep=',')
            print(f"    SAVED {year} CSV FILE")

    df = pd.concat(dfs, ignore_index=True)
    df.to_csv("processed/br-capes-colsucup-docente.csv", index=False, encoding="utf-8", sep=',')
    print(f"\nTOTAL: {len(df)} ROWS")

if __name__ == '__main__':
    print("PRE-PROCESSING CAPES DATA")

    data_files = list(glob.glob('data/*.{}'.format('csv')))
    data_files.sort()
    process_data_files(data_files)

    print("PRE-PROCESSING COMPLETED")