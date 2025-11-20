import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from gseapy import ssgsea
from gseapy.parser import read_gmt
from scipy import linalg
from scipy.special import softmax
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.preprocessing import StandardScaler

try:
    import decoupler as dc
except ImportError:  # pragma: no cover
    dc = None

HAS_DECOUPLER = dc is not None
HAS_OP = HAS_DECOUPLER and hasattr(dc, "op")
HAS_MT = HAS_DECOUPLER and hasattr(dc, "mt")
USE_DECOUPLER = HAS_DECOUPLER

try:  # Optional dependency, only needed when users pass AnnData objects.
    from anndata import AnnData
except ImportError:  # pragma: no cover
    AnnData = None

ROOT_DIR = Path(__file__).resolve().parents[2]
RAW_DATA_DIR = ROOT_DIR / "pre" / "raw_data" / "data"
DEFAULT_AML_EXPR = ROOT_DIR / "data" / "Processed_AML2_Expression_matrix.csv"
DEFAULT_CCLE_EXPR = ROOT_DIR / "data" / "Processed_CCLE_Expression_matrix.csv"
DEFAULT_ANNOTATION = RAW_DATA_DIR / "AML2_Expression.txt"
DEFAULT_PROGENY = RAW_DATA_DIR / "signatures" / "progeny_human_full.csv"
DEFAULT_DOROTHEA = RAW_DATA_DIR / "signatures" / "dorothea_hs_full.csv"
DEFAULT_HALLMARK = RAW_DATA_DIR / "signatures" / "hallmark_human.gmt"


def optional_path(path: Path) -> Optional[Path]:
    return path if path.exists() else None


def strip_version(ensembl_id: str) -> str:
    return ensembl_id.split(".")[0]


def build_gene_symbol_map(annotation_path: Path) -> Dict[str, str]:
    cols = ["stable_id", "display_label"]
    ann = pd.read_csv(annotation_path, sep="\t", usecols=lambda c: c in cols)
    ann["stable_id"] = ann["stable_id"].map(strip_version)
    ann["display_label"] = ann["display_label"].fillna("")
    ann = ann.drop_duplicates("stable_id")
    return dict(zip(ann["stable_id"], ann["display_label"]))


def prepare_expression_matrix(path: Path, id_col: str, symbol_map: Dict[str, str]) -> pd.DataFrame:
    df = pd.read_csv(path)
    gene_ids = df[id_col].map(strip_version)
    expr = df.drop(columns=[id_col])
    expr = expr.apply(pd.to_numeric, errors="coerce")
    symbols = gene_ids.map(lambda gid: symbol_map.get(gid, gid)).fillna(gene_ids)
    symbols = symbols.astype(str).str.upper()
    expr.index = symbols
    expr = expr.groupby(expr.index).mean()
    expr = expr.transpose()
    expr.index.name = "sample_id"
    return expr


def read_table_auto(path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(path, sep=None, engine="python")
    except Exception:
        return pd.read_csv(path)


def ensure_weight_column(df: pd.DataFrame, fallback_cols: Optional[Iterable[str]] = None) -> pd.DataFrame:
    if "weight" in df.columns:
        return df
    if fallback_cols:
        for col in fallback_cols:
            if col in df.columns:
                df = df.copy()
                df.rename(columns={col: "weight"}, inplace=True)
                return df
    return df


def load_progeny_resource(
    resource_path: Optional[Path],
    top: int,
    organism: str = "human",
    license_type: str = "academic",
    thr_padj: float = 0.05,
    verbose: bool = False,
) -> Optional[pd.DataFrame]:
    license_type = (license_type or "academic").lower()
    if resource_path and resource_path.exists():
        net = read_table_auto(resource_path)
        required = {"source", "target"}
        if not required.issubset(net.columns):
            raise ValueError(f"PROGENy resource must contain columns {required}")
        net = ensure_weight_column(net, fallback_cols=["coef", "beta", "b"])
        net["source"] = net["source"].astype(str)
        net["target"] = net["target"].astype(str).str.upper()
        if "weight" not in net.columns:
            raise ValueError("PROGENy resource requires a 'weight' column.")
        return net
    if HAS_OP:
        try:
            net = dc.op.progeny(
                organism=organism,
                top=top,
                thr_padj=thr_padj,
                license=license_type,
                verbose=verbose,
            )
            net = ensure_weight_column(net, fallback_cols=["coef", "beta", "b"])
            net["source"] = net["source"].astype(str)
            net["target"] = net["target"].astype(str).str.upper()
            return net
        except Exception as exc:  # pragma: no cover
            print(f"Skipping PROGENy: unable to download resource ({exc}).")
            return None
    if HAS_DECOUPLER and hasattr(dc, "get_progeny"):
        net = dc.get_progeny(organism=organism, top=top)
        net = ensure_weight_column(net, fallback_cols=["coef", "beta", "b"])
        net["source"] = net["source"].astype(str)
        net["target"] = net["target"].astype(str).str.upper()
        return net
    print("Skipping PROGENy: decoupler resource helper not available and no local file supplied.")
    return None


def load_dorothea_resource(
    resource_path: Optional[Path],
    organism: str = "human",
    license_type: str = "academic",
    levels: Optional[List[str]] = None,
    verbose: bool = False,
) -> Optional[pd.DataFrame]:
    license_type = (license_type or "academic").lower()
    if resource_path and resource_path.exists():
        net = read_table_auto(resource_path)
        required = {"source", "target"}
        if not required.issubset(net.columns):
            raise ValueError(f"DoRothEA resource must contain columns {required}")
        net = ensure_weight_column(net, fallback_cols=["mor"])
        net["source"] = net["source"].astype(str)
        net["target"] = net["target"].astype(str).str.upper()
        if "weight" not in net.columns:
            raise ValueError("DoRothEA resource requires either 'weight' or 'mor' column.")
        return net
    normalized_levels = None
    if levels:
        normalized_levels = [lvl.strip().upper() for lvl in levels if lvl.strip()]
    if HAS_OP:
        try:
            net = dc.op.dorothea(
                organism=organism,
                levels=normalized_levels,
                license=license_type,
                verbose=verbose,
            )
            net = ensure_weight_column(net, fallback_cols=["mor"])
            net["source"] = net["source"].astype(str)
            net["target"] = net["target"].astype(str).str.upper()
            return net
        except Exception as exc:  # pragma: no cover
            print(f"Skipping DoRothEA VIPER: unable to download resource ({exc}).")
            return None
    if HAS_DECOUPLER and hasattr(dc, "get_dorothea"):
        net = dc.get_dorothea(organism=organism, weight="mor", confidence=normalized_levels)
        net = ensure_weight_column(net, fallback_cols=["mor"])
        net["source"] = net["source"].astype(str)
        net["target"] = net["target"].astype(str).str.upper()
        return net
    print("Skipping DoRothEA: decoupler resource helper not available and no local file supplied.")
    return None


def load_hallmark_sets(
    gmt_path: Optional[Path],
    organism: str = "human",
    license_type: str = "academic",
    verbose: bool = False,
) -> Optional[Dict[str, List[str]]]:
    license_type = (license_type or "academic").lower()
    if gmt_path is not None:
        if not gmt_path.exists():
            raise FileNotFoundError(f"Hallmark GMT not found: {gmt_path}")
        parsed = read_gmt(str(gmt_path))
        gene_sets = {term: [g.upper() for g in genes] for term, genes in parsed.items()}
        return gene_sets
    if HAS_OP:
        try:
            hallmark_df = dc.op.hallmark(organism=organism, license=license_type, verbose=verbose)
        except Exception as exc:  # pragma: no cover
            print(f"Skipping Hallmark ssGSEA: unable to download resource ({exc}).")
            return None
        required_cols = {"geneset", "gene"}
        if not isinstance(hallmark_df, pd.DataFrame) or not required_cols.issubset(hallmark_df.columns):
            print("Skipping Hallmark ssGSEA: unexpected resource format.")
            return None
        gene_sets: Dict[str, List[str]] = {}
        for term, group in hallmark_df.groupby("geneset"):
            gene_sets[term] = [g.upper() for g in group["gene"].tolist()]
        return gene_sets
    print("Skipping Hallmark ssGSEA: no GMT provided and decoupler resource helper unavailable.")
    return None


def prefix_columns(df: pd.DataFrame, prefix: str) -> pd.DataFrame:
    df = df.copy()
    df.columns = [f"{prefix}::{col}" for col in df.columns]
    return df


def _extract_scores(result, method_label: str) -> pd.DataFrame:
    if isinstance(result, tuple):
        scores, _ = result
        return scores
    if isinstance(result, pd.DataFrame):
        return result
    if AnnData is not None and isinstance(result, AnnData):
        key = f"score_{method_label}"
        if key not in result.obsm:
            raise ValueError(f"AnnData result missing '{key}' in .obsm.")
        scores = result.obsm[key]
        if isinstance(scores, pd.DataFrame):
            return scores
        return pd.DataFrame(scores, index=result.obs_names)
    raise TypeError(f"Unsupported result type {type(result)} for method {method_label}.")


def compute_progeny_scores(expr: pd.DataFrame, net: pd.DataFrame, tmin: int) -> pd.DataFrame:
    if not USE_DECOUPLER:
        return compute_weighted_activity(expr, net, label="PROGENy")
    try:
        if HAS_MT and hasattr(dc.mt, "mlm"):
            result = dc.mt.mlm(expr, net, tmin=tmin, verbose=False)
            scores = _extract_scores(result, "mlm")
            return prefix_columns(scores, "PROGENy")
        if hasattr(dc, "run_progeny"):
            scores = dc.run_progeny(mat=expr, net=net, source="source", target="target", weight="weight")
            return prefix_columns(scores, "PROGENy")
        if hasattr(dc, "run_mlm"):
            result = dc.run_mlm(
                mat=expr,
                net=net,
                source="source",
                target="target",
                weight="weight",
                min_n=tmin,
                verbose=False,
            )
            scores = _extract_scores(result, "mlm")
            return prefix_columns(scores, "PROGENy")
    except Exception as exc:  # pragma: no cover
        print(f"Warning: PROGENy decoupler scoring failed ({exc}); falling back to weighted sum.")
    return compute_weighted_activity(expr, net, label="PROGENy")


def compute_viper_scores(expr: pd.DataFrame, regulon: pd.DataFrame, tmin: int) -> pd.DataFrame:
    if not USE_DECOUPLER:
        return compute_weighted_activity(expr, regulon, label="DoRothEA")
    try:
        if HAS_MT and hasattr(dc.mt, "viper"):
            result = dc.mt.viper(expr, regulon, tmin=tmin, verbose=False)
            scores = _extract_scores(result, "viper")
            return prefix_columns(scores, "DoRothEA")
        if hasattr(dc, "run_viper"):
            result = dc.run_viper(
                mat=expr,
                net=regulon,
                source="source",
                target="target",
                weight="weight",
                min_n=tmin,
                verbose=False,
            )
            scores = _extract_scores(result, "viper")
            return prefix_columns(scores, "DoRothEA")
    except Exception as exc:  # pragma: no cover
        print(f"Warning: VIPER scoring failed ({exc}); falling back to weighted sum.")
    return compute_weighted_activity(expr, regulon, label="DoRothEA")


def compute_weighted_activity(expr: pd.DataFrame, net: pd.DataFrame, label: str) -> pd.DataFrame:
    if net.empty:
        raise ValueError(f"{label} network is empty; cannot compute activity.")
    pivot = (
        net.pivot_table(index="target", columns="source", values="weight", aggfunc="mean")
        .fillna(0.0)
        .astype(float)
    )
    shared = pivot.index.intersection(expr.columns)
    if shared.empty:
        raise ValueError(f"No overlap between expression genes and {label} targets.")
    pivot = pivot.loc[shared]
    expr_shared = expr[shared]
    weight_mat = pivot.to_numpy()
    expr_mat = expr_shared.to_numpy()
    scores = expr_mat @ weight_mat
    counts = np.where(weight_mat != 0, 1.0, 0.0).sum(axis=0)
    counts[counts == 0] = 1.0
    scores = scores / counts
    df_scores = pd.DataFrame(scores, index=expr.index, columns=pivot.columns)
    return prefix_columns(df_scores, label)


def compute_hallmark_scores(expr: pd.DataFrame, hallmark_sets: Dict[str, List[str]], threads: int) -> pd.DataFrame:
    gene_space = expr.transpose()
    ss = ssgsea(
        data=gene_space,
        gene_sets=hallmark_sets,
        outdir=None,
        sample_norm_method="rank",
        no_plot=True,
        processes=threads,
        threads=threads,
        min_size=5,
        max_size=5000,
    )
    res = ss.res2d.pivot(index="Name", columns="Term", values="NES")
    res = res.loc[gene_space.columns]
    return prefix_columns(res, "Hallmark")


def zscore_domain(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    scaler = StandardScaler()
    z = scaler.fit_transform(df.values)
    z_df = pd.DataFrame(z, index=df.index, columns=df.columns)
    stats = pd.DataFrame({"feature": df.columns, "mean": scaler.mean_, "scale": scaler.scale_})
    return z_df, stats


def coral_align(source: pd.DataFrame, target: pd.DataFrame, reg: float = 1e-6) -> pd.DataFrame:
    src = source.values
    tgt = target.values
    mu_src = src.mean(axis=0, keepdims=True)
    mu_tgt = tgt.mean(axis=0, keepdims=True)
    cov_src = np.cov(src, rowvar=False) + reg * np.eye(src.shape[1])
    cov_tgt = np.cov(tgt, rowvar=False) + reg * np.eye(tgt.shape[1])
    whiten = linalg.fractional_matrix_power(cov_src, -0.5)
    color = linalg.fractional_matrix_power(cov_tgt, 0.5)
    aligned = (src - mu_src) @ whiten @ color + mu_tgt
    return pd.DataFrame(aligned, index=source.index, columns=source.columns)


@dataclass
class StabilityResult:
    best_k: int
    stability_table: pd.DataFrame
    best_model: KMeans


def kmeans_with_stability(z_mat: pd.DataFrame, k_values: Iterable[int], n_boot: int, sample_frac: float, random_state: int) -> StabilityResult:
    data = z_mat.values
    n_samples = data.shape[0]
    rng = np.random.default_rng(random_state)
    rows = []
    best_model = None
    best_score = (-np.inf, -np.inf)
    for k in k_values:
        label_runs = []
        for _ in range(n_boot):
            idx = rng.choice(n_samples, size=max(2, int(sample_frac * n_samples)), replace=False)
            km = KMeans(n_clusters=k, n_init=20, random_state=int(rng.integers(0, 1_000_000)))
            km.fit(data[idx])
            labels = km.predict(data)
            label_runs.append(labels)
        pair_scores = []
        for i in range(len(label_runs)):
            for j in range(i + 1, len(label_runs)):
                pair_scores.append(adjusted_rand_score(label_runs[i], label_runs[j]))
        stability = float(np.mean(pair_scores)) if pair_scores else float("nan")
        full_model = KMeans(n_clusters=k, n_init=50, random_state=random_state).fit(data)
        silhouette = float(silhouette_score(data, full_model.labels_)) if k > 1 else float("nan")
        rows.append({"k": k, "stability": stability, "silhouette": silhouette})
        score_tuple = (stability, silhouette)
        if score_tuple > best_score:
            best_score = score_tuple
            best_model = full_model
    table = pd.DataFrame(rows).sort_values("k")
    if best_model is None:
        raise RuntimeError("Failed to fit any clustering model.")
    return StabilityResult(best_k=best_model.n_clusters, stability_table=table, best_model=best_model)


def soft_assignments(model: KMeans, data: pd.DataFrame, temperature: Optional[float] = None) -> pd.DataFrame:
    distances = model.transform(data.values)
    if temperature is None:
        temperature = np.median(distances)
    logits = -distances / (temperature + 1e-8)
    probs = softmax(logits, axis=1)
    cols = [f"subtype_{i+1}" for i in range(model.n_clusters)]
    return pd.DataFrame(probs, index=data.index, columns=cols)


def save_dataframe(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix == ".parquet":
        try:
            df.to_parquet(path)
            return
        except ImportError:
            fallback = path.with_suffix(".csv")
            print(f"Warning: parquet engine unavailable; writing CSV to {fallback}.")
            df.to_csv(fallback)
            return
    else:
        df.to_csv(path)


def plot_stability_curves(table: pd.DataFrame, best_k: int, path: Path) -> None:
    if table.empty:
        return
    fig, ax1 = plt.subplots(figsize=(6, 4))
    ax1.plot(table["k"], table["stability"], marker="o", label="Bootstrap ARI", color="#1f77b4")
    ax1.set_xlabel("K")
    ax1.set_ylabel("Bootstrap ARI", color="#1f77b4")
    ax1.tick_params(axis="y", labelcolor="#1f77b4")
    ax2 = ax1.twinx()
    ax2.plot(table["k"], table["silhouette"], marker="s", label="Silhouette", color="#ff7f0e")
    ax2.set_ylabel("Silhouette", color="#ff7f0e")
    ax2.tick_params(axis="y", labelcolor="#ff7f0e")
    ax1.axvline(best_k, linestyle="--", color="gray", alpha=0.6)
    ax1.set_title("Cluster Stability Across K")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=300)
    plt.close(fig)


def plot_prototype_heatmap(prototypes: pd.DataFrame, max_features: int, path: Path) -> None:
    if prototypes.empty:
        return
    sns.set_theme(style="whitegrid")
    variances = prototypes.var(axis=0).sort_values(ascending=False)
    top_cols = variances.head(max_features).index
    data = prototypes[top_cols]
    fig, ax = plt.subplots(figsize=(max(8, len(top_cols) * 0.25), max(4, prototypes.shape[0] * 0.6)))
    sns.heatmap(data, cmap="coolwarm", center=0, ax=ax, cbar_kws={"label": "Prototype z-score"})
    ax.set_xlabel("Pathway / TF feature")
    ax.set_ylabel("Subtype")
    ax.set_title("Subtype Prototypes in Pathway Space")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=300)
    plt.close(fig)


def plot_hard_assignment_counts(pi_pat: pd.DataFrame, pi_src: pd.DataFrame, path: Path) -> None:
    if pi_pat.empty or pi_src.empty:
        return
    pat_labels = pi_pat.idxmax(axis=1).value_counts().sort_index()
    src_labels = pi_src.idxmax(axis=1).value_counts().sort_index()
    df_plot = pd.DataFrame(
        {
            "Subtype": list(pat_labels.index) + list(src_labels.index),
            "Count": list(pat_labels.values) + list(src_labels.values),
            "Domain": ["Patients"] * len(pat_labels) + ["Cell lines"] * len(src_labels),
        }
    )
    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.barplot(data=df_plot, x="Subtype", y="Count", hue="Domain", ax=ax)
    ax.set_title("Hard Assignment Counts per Subtype")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=300)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Pathway activity stratification pipeline.")
    parser.add_argument("--aml-path", type=Path, default=DEFAULT_AML_EXPR)
    parser.add_argument("--ccle-path", type=Path, default=DEFAULT_CCLE_EXPR)
    parser.add_argument("--annotation-path", type=Path, default=DEFAULT_ANNOTATION)
    parser.add_argument("--hallmark-gmt", type=Path, default=optional_path(DEFAULT_HALLMARK))
    parser.add_argument("--progeny-resource", type=Path, default=optional_path(DEFAULT_PROGENY))
    parser.add_argument("--dorothea-resource", type=Path, default=optional_path(DEFAULT_DOROTHEA))
    parser.add_argument("--organism", type=str, default="human")
    parser.add_argument("--resource-license", type=str, default="academic")
    parser.add_argument("--resource-verbose", action="store_true")
    parser.add_argument("--progeny-top", type=int, default=100)
    parser.add_argument("--progeny-thr-padj", type=float, default=0.05)
    parser.add_argument("--dorothea-levels", type=str, nargs="+", default=None)
    parser.add_argument("--min-targets", type=int, default=5)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--k-min", type=int, default=2)
    parser.add_argument("--k-max", type=int, default=6)
    parser.add_argument("--n-bootstrap", type=int, default=20)
    parser.add_argument("--sample-frac", type=float, default=0.8)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--align", choices=["none", "coral"], default="none")
    parser.add_argument("--output-dir", type=Path, default=Path("pathway_outputs"))
    parser.add_argument("--heatmap-top-pathways", type=int, default=40)
    parser.add_argument("--use-decoupler", action="store_true", help="Use decoupler engines for PROGENy/DoRothEA scoring.")
    args = parser.parse_args()
    global USE_DECOUPLER
    USE_DECOUPLER = HAS_DECOUPLER and args.use_decoupler

    if args.min_targets < 1:
        parser.error("--min-targets must be at least 1.")
    if not (0 < args.progeny_thr_padj <= 1):
        parser.error("--progeny-thr-padj must lie in (0, 1].")
    if args.heatmap_top_pathways < 1:
        parser.error("--heatmap-top-pathways must be at least 1.")

    symbol_map = build_gene_symbol_map(args.annotation_path)
    aml_expr = prepare_expression_matrix(args.aml_path, "stable_id", symbol_map)
    ccle_expr = prepare_expression_matrix(args.ccle_path, "gene_id", symbol_map)

    progeny_net = load_progeny_resource(
        args.progeny_resource,
        top=args.progeny_top,
        organism=args.organism,
        license_type=args.resource_license,
        thr_padj=args.progeny_thr_padj,
        verbose=args.resource_verbose,
    )
    dorothea_net = load_dorothea_resource(
        args.dorothea_resource,
        organism=args.organism,
        license_type=args.resource_license,
        levels=args.dorothea_levels,
        verbose=args.resource_verbose,
    )
    hallmark_sets = load_hallmark_sets(
        args.hallmark_gmt,
        organism=args.organism,
        license_type=args.resource_license,
        verbose=args.resource_verbose,
    )

    pathway_scores = {}
    for domain_name, expr in [("patients", aml_expr), ("cell_lines", ccle_expr)]:
        components = []
        if progeny_net is not None:
            components.append(compute_progeny_scores(expr, progeny_net, tmin=args.min_targets))
        if hallmark_sets is not None:
            components.append(compute_hallmark_scores(expr, hallmark_sets, args.threads))
        if dorothea_net is not None:
            components.append(compute_viper_scores(expr, dorothea_net, tmin=args.min_targets))
        if not components:
            raise RuntimeError("No pathway/TF activities could be computed. Provide at least one resource.")
        combined = pd.concat(components, axis=1)
        pathway_scores[domain_name] = combined
        save_dataframe(combined, args.output_dir / f"{domain_name}_pathway_scores.parquet")

    z_patients, stats_pat = zscore_domain(pathway_scores["patients"])
    z_cell_lines, stats_ccle = zscore_domain(pathway_scores["cell_lines"])

    if args.align == "coral":
        z_cell_lines = coral_align(z_cell_lines, z_patients)

    save_dataframe(z_patients, args.output_dir / "Z_patients.parquet")
    save_dataframe(z_cell_lines, args.output_dir / "Z_cell_lines.parquet")
    save_dataframe(stats_pat, args.output_dir / "scaler_patients.csv")
    save_dataframe(stats_ccle, args.output_dir / "scaler_cell_lines.csv")

    k_values = range(args.k_min, args.k_max + 1)
    stability = kmeans_with_stability(z_patients, k_values, args.n_bootstrap, args.sample_frac, args.random_state)
    prototypes = pd.DataFrame(stability.best_model.cluster_centers_, columns=z_patients.columns)
    prototypes.index = [f"subtype_{i+1}" for i in range(stability.best_model.n_clusters)]
    save_dataframe(prototypes, args.output_dir / "P_path.csv")
    save_dataframe(stability.stability_table, args.output_dir / "stability_summary.csv")

    pi_pat = soft_assignments(stability.best_model, z_patients)
    pi_src = soft_assignments(stability.best_model, z_cell_lines)
    save_dataframe(pi_pat, args.output_dir / "pi_pat.csv")
    save_dataframe(pi_src, args.output_dir / "pi_src.csv")

    figures_dir = args.output_dir / "figures"
    plot_stability_curves(stability.stability_table, stability.best_k, figures_dir / "stability_curves.png")
    plot_prototype_heatmap(prototypes, args.heatmap_top_pathways, figures_dir / "prototype_heatmap.png")
    plot_hard_assignment_counts(pi_pat, pi_src, figures_dir / "assignment_counts.png")

    alignment_note = "CORAL alignment applied." if args.align == "coral" else "No domain alignment."
    metrics = {
        "best_k": stability.best_k,
        "stability": stability.stability_table.loc[stability.stability_table["k"] == stability.best_k, "stability"].item(),
        "silhouette": stability.stability_table.loc[stability.stability_table["k"] == stability.best_k, "silhouette"].item(),
        "alignment": alignment_note,
    }
    (args.output_dir / "artifacts.json").write_text(json.dumps(metrics, indent=2))

    print(f"Selected K={metrics['best_k']} with stability={metrics['stability']:.3f}, silhouette={metrics['silhouette']:.3f}.")
    print(alignment_note)
    print(f"Artifacts saved to {args.output_dir.resolve()}")


if __name__ == "__main__":  # pragma: no cover
    main()
