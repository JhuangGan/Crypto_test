from tqdm import tqdm
import pandas as pd
import requests
import json
import time
import numpy as np
import os

def make_url(stamp_start, stamp_end, crypto_id, convert_id=2781):
    return f"https://api.coinmarketcap.com/data-api/v3.1/cryptocurrency/historical?id={crypto_id}&convertId={convert_id}&timeStart={stamp_start}&timeEnd={stamp_end}&interval=1d"

def make_time_range(start, end):
    start = pd.to_datetime(start) - pd.offsets.Day(1)
    end = pd.to_datetime(end)
    date_range = pd.date_range(start, end, freq="YS").to_list()
    date_range = [start] + date_range + [end]
    date_range = np.array(sorted(list(set(date_range))))
    starts = date_range[:-1]
    starts = [int(i.timestamp()) for i in starts]
    ends = np.roll(date_range, -1)[:-1]
    ends = [int(i.timestamp()) for i in ends]
    return list(zip(starts, ends))

def parse_json(data):
    if "data" not in data or "quotes" not in data["data"]:
        return pd.DataFrame(), "Unknown"
    
    quotes = data["data"]["quotes"]
    if len(quotes) == 0:
        return pd.DataFrame(), data["data"].get("name", "Unknown")
    
    code = data["data"]["id"]
    name = data["data"]["name"]
    symbol = data["data"]["symbol"]
    
    record_ls = []
    for record in quotes:
        record_ls.append(record["quote"])
    
    df = pd.DataFrame(record_ls)
    df["timestamp"] = pd.to_datetime(df["timestamp"].str[:10], utc=True)
    df["name"] = name
    df["code"] = code
    df["symbol"] = symbol
    
    return df, name

def validate_data(df, start_date, end_date):
    if df.empty:
        print("Empty DataFrame")
        return False
    
    start_utc = pd.to_datetime(start_date, utc=True).date()
    end_utc = pd.to_datetime(end_date, utc=True).date()
    
    first_date = df['timestamp'].iloc[0].date()
    last_date = df['timestamp'].iloc[-1].date()
    
    if (first_date != start_utc) or (last_date != end_utc):
        print(f"Expected: {start_utc} - {end_utc}, Got: {first_date} - {last_date}")
        return False
    
    expected_days = (pd.to_datetime(end_date) - pd.to_datetime(start_date)).days + 1
    if len(df) < expected_days:
        print(f"Expected: {expected_days} days, Got: {len(df)} days")
        return False
    
    # 加入市值规模验证，仅保留市值大于1m的数据
    if df['marketCap'].min() < 1e6:
        print(f"Market cap too small: {df['market_cap'].min()}")
        return False

    return True


if __name__ == "__main__":

    success_count = 0
    if not os.path.exists("./data"):
        os.makedirs("./data")

    for code in tqdm(range(1, 2000), desc="Crawling"):
        if success_count >= 100:
            break
        try:
            time_range = make_time_range("2020-03-05", "2025-03-05")
            df_ls = []
            for start, end in time_range:
                url = make_url(start, end, code)
                r = requests.get(url, timeout=10)
                r.raise_for_status()
                data = json.loads(r.text)
                df_chunk, name = parse_json(data)
                if not df_chunk.empty:
                    df_ls.append(df_chunk)
                time.sleep(2)  
            
            if not df_ls:
                continue
                
            res = pd.concat(df_ls).sort_values('timestamp')
            
            # 数据验证
            if validate_data(res, "2020-03-05", "2025-03-05"):
                res.to_csv(f"./data/code{code}_{name}.csv", index=False)
                success_count += 1
                print(f"\n成功保存: code{code}_{name}")
            else:
                print(f"\n数据验证失败: code{code}_{name}")
                
        except (requests.exceptions.RequestException, 
                json.JSONDecodeError,
                KeyError,
                pd.errors.EmptyDataError) as e:
            print(f"\n跳过 code{code}: {str(e)[:50]}...")
            continue
