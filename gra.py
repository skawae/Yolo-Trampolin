import pandas as pd
import numpy as np

def analyze_somersault_influence(csv_path):
    # 1. データの読み込みと前処理
    df = pd.read_csv(csv_path)
    df = df.replace("-", 0.0)
    
    numeric_cols = ["位置_X(m)", "位置_Y(m)", "前回からの移動_X(m)", "前回からの移動_Y(m)", "総移動距離(m)", "膝関節角度", "腕上げ角度", "頭部傾き角度", "腰上半身角度"]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col])
        
    # 2. 「現在の着地姿勢・ブレ」と「次の宙返りの移動距離」のマッピング
    # n回目の着地要素を原因(X)、n+1回目の総移動距離を結果(Y)とする
    analysis_data = pd.DataFrame()
    
    # 原因となる要素（比較系列）
    analysis_data["直前着地_ブレX"] = df["前回からの移動_X(m)"].abs().iloc[:-1].values
    analysis_data["直前着地_ブレY"] = df["前回からの移動_Y(m)"].abs().iloc[:-1].values
    analysis_data["直前着地_膝角度"] = df["膝関節角度"].iloc[:-1].values
    analysis_data["直前着地_腰角度"] = df["腰上半身角度"].iloc[:-1].values
    analysis_data["直前着地_頭傾き"] = df["頭部傾き角度"].abs().iloc[:-1].values
    
    # 結果となる要素（基準系列）：次の跳躍の総移動距離
    analysis_data["次宙返りの移動距離"] = df["総移動距離(m)"].shift(-1).dropna().values

    print("--- [Step 1] 分析用マッピングデータ (1行スライド済) ---")
    print(analysis_data)
    print("\n" + "="*60 + "\n")

    # 3. データの正規化 (すべての項目を 0 〜 1 に変換)
    # 基準系列も含めて、要素の最大・最小でスケーリングします（Mean-standardization等も使われますが、ここでは最大最小を利用）
    scaled_df = pd.DataFrame()
    for col in analysis_data.columns:
        max_val = analysis_data[col].max()
        min_val = analysis_data[col].min()
        if max_val == min_val:
            scaled_df[col] = 1.0
        else:
            scaled_df[col] = (analysis_data[col] - min_val) / (max_val - min_val)

    # 4. グレーリレーショナル分析 (GRA) による影響度の計算
    # 基準系列 (Target) と 比較系列 (Factors) に分離
    target_series = scaled_df["次宙返りの移動距離"].values
    factors_df = scaled_df.drop(columns=["次宙返りの移動距離"])
    
    # 識別係数
    rho = 0.5
    
    # 各要素とターゲットとの絶対差 (Δ) の行列を計算
    diff_matrix = np.zeros(factors_df.shape)
    for i, col in enumerate(factors_df.columns):
        diff_matrix[:, i] = np.abs(factors_df[col].values - target_series)
        
    delta_min = diff_matrix.min()
    delta_max = diff_matrix.max()
    
    # グレーリレーショナル係数 (GRC) の計算
    grc_matrix = (delta_min + rho * delta_max) / (diff_matrix + rho * delta_max)
    
    # 各要素の全試行における平均値を「グレーリレーショナル度 (GRG)」とする
    # この値が 1 に近いほど、ターゲット（宙返りの移動距離）との関係性・影響度が強い
    grg_scores = np.mean(grc_matrix, axis=0)
    
    # 5. 結果のランキング化
    influence_df = pd.DataFrame({
        "着地要素 (因子)": factors_df.columns,
        "影響度スコア (GRG)": grg_scores
    })
    influence_df["影響度順位"] = influence_df["影響度スコア (GRG)"].rank(ascending=False, method="min").astype(int)
    influence_df = influence_df.sort_values(by="影響度順位")
    
    print("--- [Step 2] 宙返りの移動距離に対する『着地要素の影響度ランキング』 ---")
    print(influence_df.to_string(index=False))
    
    # 結果の保存
    influence_df.to_csv("somersault_influence_output.csv", index=False, encoding="utf-8-sig")
    print("\n【保存完了】影響度分析結果を 『somersault_influence_output.csv』 に出力しました。")

if __name__ == "__main__":
    csv_path = "kawae0428-1-2/landing_analysis_report.csv"
    analyze_somersault_influence(csv_path)