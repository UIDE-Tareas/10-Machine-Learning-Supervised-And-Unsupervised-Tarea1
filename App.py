# UTILIDADES PARA GESTIÓN DE DEPENDENCIAS, INFORMACIÓN DEL ENTORNO Y FUNCIONES AUXILIARES

import sys
import subprocess
import os
from pathlib import Path
from enum import Enum
import zipfile
import warnings
from datetime import datetime
from typing import Optional
from dataclasses import dataclass
from typing import Any
from typing import Protocol

# Libs a instalar
LIBS = [
    "numpy",
    "pandas",
    "matplotlib",
    "seaborn",
    "scikit-learn",
    "requests",
    "wcwidth",
    "kneed",
]

RANDOM_STATE = 216
TRAIN_RATIO = 0.80
TEST_RATIO = 0.20
POLYNOMIAL_DEGREE = 2
RIDGE_ALPHA = 1.0
CORRELATION_THRESHOLD = 0.30
SHOW_PLOTS = True
SAVE_PLOTS = True
DATASET_URL = "https://www.kaggle.com/datasets/nikhil7280/student-performance-multiple-linear-regression"
DATASET_ARCHIVE_RAW_URL = "https://raw.githubusercontent.com/UIDE-Tareas/10-Machine-Learning-Supervised-And-Unsupervised-Tarea1/main/Data/archive.zip"
DATASET_ARCHIVE_FILENAME = "archive.zip"
DATASET_CSV_FILENAME = "Student_Performance.csv"
TARGET_COLUMN = "PerformanceIndex"
TEMP_DIR = "Temp"
RESULTS_DIR = Path("Results")
RUN_RESULTS_DIR = RESULTS_DIR / datetime.now().strftime("%Y%m%d-%H%M%S")


class ConsoleColor(Enum):
    RED = "\033[91m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    BLUE = "\033[94m"
    MAGENTA = "\033[95m"
    CYAN = "\033[96m"
    WHITE = "\033[97m"
    RESET = "\033[0m"


def PrintColor(message: str, color: ConsoleColor) -> str:
    RESET = ConsoleColor.RESET.value
    return f"{color.value}{message}{RESET}"


def ShowMessage(
    message: str, title: str, icon: str, color: ConsoleColor, end: str = "\n"
):
    colored_title = PrintColor(icon + f"  " + title.upper() + ":", color)
    print(f"{colored_title} {message}", end=end)


def ShowInfoMessage(
    message: str, title: str = "Info", icon: str = "ℹ️", end: str = "\n"
):
    ShowMessage(message, title, icon, ConsoleColor.CYAN, end)


def ShowSuccessMessage(
    message: str, title: str = "Success", icon: str = "✅", end: str = "\n"
):
    ShowMessage(message, title, icon, ConsoleColor.GREEN, end)


def ShowErrorMessage(
    message: str, title: str = "Error", icon: str = "❌", end: str = "\n"
):
    ShowMessage(message, title, icon, ConsoleColor.RED, end)


def ShowWarningMessage(
    message: str, title: str = "Warning", icon: str = "⚠️", end: str = "\n"
):
    ShowMessage(message, title, icon, ConsoleColor.YELLOW, end)


# Funcion para ejecutar comandos
def RunCommand(
    commandList: list[str], printCommand: bool = True, printError: bool = True
) -> subprocess.CompletedProcess[str]:
    print("⏳", " ".join(commandList))

    if printCommand:
        proc = subprocess.Popen(
            commandList,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
            universal_newlines=True,
        )

        out_lines: list[str] = []
        assert proc.stdout is not None
        for line in proc.stdout:
            print(line, end="")
            out_lines.append(line)

        proc.wait()
        err_text = ""
        if proc.stderr is not None:
            err_text = proc.stderr.read() or ""

        if proc.returncode != 0 and printError and err_text:
            ShowErrorMessage(err_text, "", end="")
            # print(err_text, end="")

        return subprocess.CompletedProcess(
            args=commandList,
            returncode=proc.returncode,
            stdout="".join(out_lines),
            stderr=err_text,
        )

    else:
        result = subprocess.run(
            commandList, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
        )
        if result.returncode != 0 and printError and result.stderr:
            ShowErrorMessage(result.stderr, "", end="")
            # print(result.stderr, end="")
        return result


# Función para instalar las dependencias
def InstallDeps(libs: Optional[list[str]] = None):
    print("ℹ️ Installing deps.")
    printCommand = False
    printError = True
    RunCommand(
        [sys.executable, "-m", "pip", "install", "--upgrade", "pip"],
        printCommand=printCommand,
        printError=printError,
    )
    if libs is None or libs.count == 0:
        print("No hay elementos a instalar.")
    else:
        RunCommand(
            [sys.executable, "-m", "pip", "install", *libs],
            printCommand=printCommand,
            printError=printError,
        )
        print("Deps installed.")
    print()


# Función para mostrar info el ambiente de ejecución
def ShowEnvironmentInfo():
    print("ℹ️  Environment Info:")
    print("Python Version:", sys.version)
    print("Platform:", sys.platform)
    print("Executable Path:", sys.executable)
    print("Current Working Directory:", os.getcwd())
    print("VIRTUAL_ENV:", os.environ.get("VIRTUAL_ENV"))
    print("sys.prefix:", sys.prefix)
    print("sys.base_prefix:", sys.base_prefix)
    print()


InstallDeps(LIBS)
ShowEnvironmentInfo()

def ClearConsole():
    if os.name == "nt":
        RunCommand(["cmd", "/c", "cls"], printCommand=False, printError=False)
    else:
        RunCommand(["clear"], printCommand=False, printError=False)



# Third-party imports after dependency installation.
import requests
import pandas
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from matplotlib.axes import Axes
from matplotlib.figure import Figure
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.preprocessing import MaxAbsScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import PowerTransformer
from sklearn.preprocessing import QuantileTransformer
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import StandardScaler


@dataclass(frozen=True)
class BoxStyle:
    TL: str
    TR: str
    BL: str
    BR: str
    H: str
    V: str

class TitleBoxLineStyle(Enum):
    SIMPLE = BoxStyle("┌", "┐", "└", "┘", "─", "│")
    DOUBLE = BoxStyle("╔", "╗", "╚", "╝", "═", "║")
    ROUNDED = BoxStyle("╭", "╮", "╰", "╯", "─", "│")
    HEAVY = BoxStyle("┏", "┓", "┗", "┛", "━", "┃")
    ASCII = BoxStyle("+", "+", "+", "+", "-", "|")
    DOUBLE_BOLD = BoxStyle("╔", "╗", "╚", "╝", "╬", "║")
    BLOCK = BoxStyle("█", "█", "█", "█", "█", "█")
    HEAVY_CROSS = BoxStyle("╒", "╕", "╘", "╛", "╪", "┃")
    METAL = BoxStyle("╞", "╡", "╘", "╛", "═", "║")


# Función para mostrar un título con recuadro
def ShowTitleBox(
    text: str,
    max_len: int = 100,
    boxLineStyle: TitleBoxLineStyle = TitleBoxLineStyle.SIMPLE,
    color: ConsoleColor = ConsoleColor.CYAN,
):
    try:

        def vislen(s: str) -> int:
            from wcwidth import wcswidth as _w

            n = _w(s)
            return n if n >= 0 else len(s)

    except Exception:

        def vislen(s: str) -> int:
            return len(s)

    pad = 1
    tlen = vislen(text)
    inner = max(max_len, tlen)
    left = (inner - tlen) // 2
    right = inner - tlen - left

    top = f"{boxLineStyle.value.TL}{boxLineStyle.value.H * (inner + 2 * pad)}{boxLineStyle.value.TR}"
    mid = f"{boxLineStyle.value.V}{' ' * pad}{' ' * left}{text}{' ' * right}{' ' * pad}{boxLineStyle.value.V}"
    bot = f"{boxLineStyle.value.BL}{boxLineStyle.value.H * (inner + 2 * pad)}{boxLineStyle.value.BR}"
    print(PrintColor("\n".join([top, mid, bot]), color))


# Función para descargar un archivo
def DownloadFile(uri: str, filename: str, overwrite: bool = False, timeout: int = 20, printInfo: bool = True):
    dest = Path(filename).resolve()
    if dest.exists() and dest.is_file() and dest.stat().st_size > 0 and not overwrite:
        if printInfo:
            print(
                f'✅ Ya existe: "{dest}". No se descarga (use overwrite=True para forzar).'
            )
        return
    if dest.parent and not dest.parent.exists():
        dest.parent.mkdir(parents=True, exist_ok=True)
    if printInfo:
        print(f'ℹ️ Descargando "{uri}" → "{dest}"')
    try:
        with requests.get(uri, stream=True, timeout=timeout) as resp:
            resp.raise_for_status()
            tmp = dest.with_suffix(dest.suffix + ".part")
            with open(tmp, "wb") as f:
                for chunk in resp.iter_content(chunk_size=1024 * 64):
                    if chunk:  # filtra keep-alive chunks
                        f.write(chunk)
            tmp.replace(dest)
        if printInfo: 
            print(f'✅ Archivo "{dest}" descargado exitosamente.')
    except requests.exceptions.RequestException as e:
        print(f"❌ Error al descargar: {e}")


# Función para descomprimir un archivo zip
def UnzipFile(filename: str, outputDir: str):
    print(f'ℹ️ Descomprimiendo "{filename}" en "{outputDir}"')
    try:
        with zipfile.ZipFile(filename, "r") as zip_ref:
            zip_ref.extractall(outputDir)
        print(f"Descomprimido en: {os.path.abspath(outputDir)}")
    except Exception as e:
        print(f"Error: {e}")


def CreateRunResultsDir(resultsDir: Path = RUN_RESULTS_DIR) -> Path:
    resultsDir.mkdir(parents=True, exist_ok=True)
    ShowSuccessMessage(f'Directorio de resultados: "{resultsDir.resolve()}".')
    return resultsDir


def ShowDatasetUrl(title: str = "Stage 1 - Dataset URL", url: str = DATASET_URL):
    ShowTitleBox(
        title,
        boxLineStyle=TitleBoxLineStyle.DOUBLE,
        color=ConsoleColor.GREEN,
    )
    print(f"Dataset URL: {url}")


def DownloadDatasetArchive(title: str = "Stage 2 - Download Dataset"):
    ShowTitleBox(
        title,
        boxLineStyle=TitleBoxLineStyle.DOUBLE,
        color=ConsoleColor.GREEN,
    )
    archivePath = Path(TEMP_DIR) / DATASET_ARCHIVE_FILENAME
    DownloadFile(
        DATASET_ARCHIVE_RAW_URL,
        str(archivePath),
        overwrite=False,
    )
    return archivePath


def UnzipDatasetArchive(title: str = "Stage 2 - Download Dataset"):
    archivePath = DownloadDatasetArchive(title)
    csvPath = Path(TEMP_DIR) / DATASET_CSV_FILENAME
    if csvPath.exists() and csvPath.is_file() and csvPath.stat().st_size > 0:
        print(f'✅ Ya existe: "{csvPath.resolve()}". No se descomprime nuevamente.')
        return csvPath
    UnzipFile(str(archivePath), TEMP_DIR)
    return csvPath


def LoadRawDataset(title: str = "Stage 3 - Read Dataset") -> pandas.DataFrame:
    ShowTitleBox(
        title,
        boxLineStyle=TitleBoxLineStyle.DOUBLE,
        color=ConsoleColor.GREEN,
    )
    csvPath = Path(TEMP_DIR) / DATASET_CSV_FILENAME
    dfRaw = pandas.read_csv(csvPath)
    ShowSuccessMessage(
        f'Dataset cargado desde "{csvPath.resolve()}". Filas: {dfRaw.shape[0]}, Columnas: {dfRaw.shape[1]}'
    )
    return dfRaw


def ShowRawDatasetInfo(
    dfRaw: pandas.DataFrame,
    stageTitle: str = "Stage 4 - Inspect DataFrame",
    dfTitle: str = "dfRaw",
):
    ShowTitleBox(
        stageTitle,
        boxLineStyle=TitleBoxLineStyle.DOUBLE,
        color=ConsoleColor.GREEN,
    )
    ShowDfInfo(dfRaw, f"Información - {dfTitle}")
    ShowDfNanValues(dfRaw, f"Valores nulos - {dfTitle}")
    ShowDfDuplicates(dfRaw, f"Duplicados - {dfTitle}")
    ShowDfStats(dfRaw, f"Estadística descriptiva - {dfTitle}")
    ShowDfHead(dfRaw, f"Primeras 10 filas - {dfTitle}", headQty=10)


def CleanDataFrame(
    dfRaw: pandas.DataFrame,
    stageTitle: str = "Stage 5 - Clean DataFrame",
    dfTitle: str = "dfClean",
) -> pandas.DataFrame:
    ShowTitleBox(
        stageTitle,
        boxLineStyle=TitleBoxLineStyle.DOUBLE,
        color=ConsoleColor.GREEN,
    )
    dfClean = dfRaw.copy()
    dfClean = pandas.get_dummies(dfClean, drop_first=True)
    ShowSuccessMessage("One-Hot Encoding aplicado con drop_first=True.")
    boolColumns = dfClean.select_dtypes(include="bool").columns
    dfClean[boolColumns] = dfClean[boolColumns].astype(int)
    ShowSuccessMessage("Columnas booleanas convertidas a números 0/1.")
    dfClean = RemoveDfDuplicates(dfClean)
    ShowSuccessMessage("Filas duplicadas eliminadas.")
    dfClean = dfClean.dropna().reset_index(drop=True)
    ShowSuccessMessage("Filas con valores nulos eliminadas.")
    dfClean = NormalizeColumnNames(dfClean)
    ShowSuccessMessage("Nombres de columnas normalizados.")
    ShowDfInfo(dfClean, f"Información - {dfTitle}")
    ShowDfNanValues(dfClean, f"Valores nulos - {dfTitle}")
    ShowDfDuplicates(dfClean, f"Duplicados - {dfTitle}")
    ShowDfStats(dfClean, f"Estadística descriptiva - {dfTitle}")
    ShowDfHead(dfClean, f"Primeras 10 filas - {dfTitle}", headQty=10)
    return dfClean


def PlotTargetDistribution(
    df: pandas.DataFrame,
    targetColumn: str = TARGET_COLUMN,
    stageTitle: str = "Stage 6 - Target Distribution",
    resultsDir: Path = RUN_RESULTS_DIR,
    showPlot: bool = SHOW_PLOTS,
    savePlot: bool = SAVE_PLOTS,
):
    ShowTitleBox(
        stageTitle,
        boxLineStyle=TitleBoxLineStyle.DOUBLE,
        color=ConsoleColor.GREEN,
    )

    if targetColumn not in df.columns:
        raise ValueError(f'No existe la columna objetivo "{targetColumn}" en el DataFrame.')

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.canvas.manager.set_window_title(f"{targetColumn} - Target Distribution")

    sns.histplot(df[targetColumn], kde=True, ax=axes[0])
    axes[0].set_title(f"{targetColumn} - Distribution")
    axes[0].set_xlabel(targetColumn)
    axes[0].set_ylabel("Frequency")

    sns.boxplot(x=df[targetColumn], ax=axes[1])
    axes[1].set_title(f"{targetColumn} - Boxplot")
    axes[1].set_xlabel(targetColumn)

    plt.tight_layout()
    outputPath = resultsDir / f"{targetColumn}_Distribution.png"
    if savePlot:
        outputPath.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(outputPath, dpi=150)
        ShowSuccessMessage(f'Gráfico guardado en "{outputPath.resolve()}".')
    if showPlot:
        plt.show()
    else:
        plt.close(fig)


#### ███████████████████████████████████████████████ -------------


# UTILIDADES PARA ANÁLISIS Y MANIPULACIÓN DE DATAFRAMES

warnings.filterwarnings("ignore")

# Configurar opciones de Pandas
pd.set_option("display.float_format", "{:.2f}".format)
pandas.set_option("display.max_rows", None)
pandas.set_option("display.max_columns", None)


def PrintData(value: Any = ""):
    if isinstance(value, pandas.DataFrame):
        print(value.to_string())
    elif isinstance(value, pandas.Series):
        print(value.to_string())
    else:
        print(value)


# Función para mostrar la información del DataFrame.
def ShowDfInfo(df: pandas.DataFrame, title):
    print(f"ℹ️ INFO {title} ℹ️")
    df.info()
    print()


# Función para mostrar las n primeras filas del DataFrame.
def ShowDfHead(df: pandas.DataFrame, title: str, headQty=10):
    print(f"ℹ️ {title}: Primeros {headQty} elementos.")
    PrintData(df.head(headQty))
    print()


# Función para mostrar las n últimas filas del DataFrame.
def ShowDfTail(df: pandas.DataFrame, title: str, tailQty=10):
    print(f"ℹ️ {title}: Últimos {tailQty} elementos.")
    PrintData(df.tail(tailQty))
    print()


# Mostrar el tamaño del DataFrame
def ShowDfShape(df: pandas.DataFrame, title: str):
    print(f"ℹ️ {title} - Tamaño de los datos")
    print(f"{df.shape[0]} filas x {df.shape[1]} columnas")
    print()


# Función para mostrar la estadística descriptiva de todas las columnas del DataFrame, por tipo de dato.
def ShowDfStats(df: pandas.DataFrame, title: str = ""):
    print(f"ℹ️ Estadística descriptiva - {title}")
    numeric_cols = df.select_dtypes(include="number")
    if not numeric_cols.empty:
        print("    🔢 Columnas numéricas".upper())
        numeric_desc = (
            numeric_cols.describe().round(2).T
        )  # Transpuesta para añadir columna
        numeric_desc["var"] = numeric_cols.var(numeric_only=True).round(2)
        PrintData(numeric_desc.T)
    non_numeric_cols = df.select_dtypes(
        include=["boolean", "string", "category", "object"]
    )
    if not non_numeric_cols.empty:
        print("    🔡 Columnas no numéricas".upper())
        non_numeric_desc = non_numeric_cols.describe()
        PrintData(non_numeric_desc)
    datetime_cols = df.select_dtypes(include=["datetime", "datetimetz"])
    if not datetime_cols.empty:
        print("    📅 Columnas fechas".upper())
        datetime_desc = datetime_cols.describe()
        PrintData(datetime_desc)

# Función para mostrar los duplicados de un DataFrame, agrupados por todas las columnas y ordenados por cantidad de ocurrencias.
def ShowDfDuplicates(
    df: pandas.DataFrame,
    title: str,
    qty: int = 10
):
    print(f"ℹ️ Duplicados en el DataFrame - {title}")

    dup_mask = df.duplicated()
    dup_df = df[dup_mask]
    dup_count = dup_df.shape[0]

    print(f"Cantidad de filas duplicadas: {dup_count}")

    if dup_count == 0:
        print("No se encontraron duplicados.")
        print()
        return

    print(f"Mostrando hasta {qty} duplicados:")
    PrintData(dup_df.head(qty))

    remaining = dup_count - qty
    if remaining > 0:
        print(f"⚠️ Existen {remaining} filas duplicadas adicionales no mostradas.")

    print()

# Función para mostrar los valores únicos de cada columna, con un límite opcional para no mostrar demasiados valores.
def ShowDfUniqueValues(df: pandas.DataFrame, title: str, qty: int = 30):
    print(f"ℹ️ Valores únicos por columna - {title}")

    for col in df.columns:
        unique_vals = df[col].dropna().unique()
        count = len(unique_vals)

        print(f"Columna: {col}")
        print(f"Cantidad de valores únicos: {count}")

        if count <= qty:
            PrintData(sorted(unique_vals))
        else:
            print(f"Más de {qty} valores únicos. Mostrando primeros {qty}:")
            PrintData(sorted(unique_vals[:qty]))

        print()

# Función para mostrar una visión general completa del DataFrame
def ShowFullDfOverview(df, title, headQty=5, tailQty=5, duplicatesqty = 10, uniqueqty = 30):
    ShowDfInfo(df, title)
    ShowDfStats(df, title)
    ShowDfShape(df, title)
    ShowDfDuplicates(df, title, qty=duplicatesqty)
    ShowDfUniqueValues(df, title, qty=uniqueqty)
    ShowDfHead(df, title, headQty=headQty)
    ShowDfTail(df, title, tailQty=tailQty)


# Función para mostrar los valores nulos o NaN de cada columna en un DataFrame
def ShowDfNanValues(df: pandas.DataFrame, title: str):
    print(f"ℹ️ Contador de valores Nulos - {title}")
    nulls_count = df.isnull().sum()
    nulls_df = nulls_count.reset_index()
    nulls_df.columns = ["Columna", "Cantidad_Nulos"]
    PrintData(nulls_df)
    print()


# Tipos de correlación
class CorrelationType(Enum):
    ALL = "all"
    STRONG = "strong"
    WEAK = "weak"


# Muestra las correlaciones completas, débiles y fuertes.
def ShowDfCorrelation(
    df: pandas.DataFrame,
    title: str,
    fig: Optional[Figure] = None,
    ax: Optional[Axes] = None,
    level: CorrelationType = CorrelationType.ALL,
    umbral: float = 0.6,
    showTable: bool = False,
    figsize: tuple = (8, 6),
    annotate: bool = True,
    outputPath: Optional[Path] = None,
    showPlot: bool = SHOW_PLOTS,
    savePlot: bool = SAVE_PLOTS,
):
    if fig is None or ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    fig.canvas.manager.set_window_title(f"{title} - {level.name} Correlation")

    print(f"ℹ️ {title.upper()} - MATRIZ DE CORRELACIÓN ({level.name})")

    corr = df.select_dtypes(include="number").corr()

    if level == CorrelationType.STRONG:
        corr = corr.where(np.abs(corr) >= umbral)
    elif level == CorrelationType.WEAK:
        corr = corr.where((np.abs(corr) < umbral) | (corr == 1))
    elif level != CorrelationType.ALL:
        raise ValueError(f"Invalid level: {level}")

    # ✅ Mostrar diagonal (1) y triángulo inferior
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)

    sns.heatmap(
        corr,
        mask=mask,
        cmap="coolwarm",
        vmin=-1,
        vmax=1,
        annot=annotate,
        fmt=".2f",
        linewidths=0.5,
        cbar_kws={"label": "Coeficiente de correlación"},
        ax=ax
    )

    subtitle = (
        "Completa"
        if level == CorrelationType.ALL
        else f"Strong (|r| ≥ {umbral})"
        if level == CorrelationType.STRONG
        else f"Weak (|r| < {umbral})"
    )

    ax.set_title(
        f"Matriz de correlación ({subtitle})",
        fontsize=12,
        pad=15
    )

    ax.tick_params(axis="x", rotation=90)
    ax.tick_params(axis="y", rotation=0)

    plt.tight_layout()
    if savePlot and outputPath is not None:
        outputPath.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(outputPath, dpi=150)
        ShowSuccessMessage(f'Gráfico guardado en "{outputPath.resolve()}".')
    if showPlot:
        plt.show()
    else:
        plt.close(fig)

    if showTable:
        PrintData(corr.round(3))

    return fig, corr

# Función para normalizar los nombres de las columnas de un DataFrame (strip, title, remove spaces and underscores).
def NormalizeColumnNames(df: pandas.DataFrame) -> pandas.DataFrame:
    df.columns = [
        col.strip().title().replace(" ", "").replace("_", "") for col in df.columns
    ]
    return df

# Función para eliminar columnas de un DataFrame, si existen, con opción inplace o retornando una copia.
def DropColumns(
    df: pandas.DataFrame, toDrop: list[str], inplace: bool = False
) -> pandas.DataFrame:
    if not toDrop:
        return df
    if inplace:
        df.drop(columns=df.columns.intersection(toDrop), inplace=True)
        return df
    else:
        return df.drop(columns=df.columns.intersection(toDrop))


# Para almacenar los datos del dataset
@dataclass
class Dataset:
    X: pandas.DataFrame
    y: pandas.DataFrame


# Para almacenar los datos de split del dataset.
@dataclass
class DatasetSplit:
    Train: Dataset
    Test: Dataset


# Muestra el head de cada componente del split.
def ShowDatasetSplitHead(split: DatasetSplit, title: str, headQty: int = 5):
    ShowDfHead(split.Train.X, f"{title} - X Train", headQty)
    ShowDfHead(split.Train.y, f"{title} - y Train", headQty)
    ShowDfHead(split.Test.X, f"{title} - X Test", headQty)
    ShowDfHead(split.Test.y, f"{title} - y Test", headQty)


# Muestra la información del Dataset
def ShowDatasetInfo(data: Dataset, title):
    tAux = title
    title = f"{tAux} - Caracteristicas - X"
    ShowDfInfo(data.X, title)
    ShowDfShape(data.X, title)
    ShowDfStats(data.X, title)
    ShowDfNanValues(data.X, title)
    ShowDfHead(data.X, title)
    ShowDfTail(data.X, title)
    title = f"{tAux} - Características - y"
    ShowDfInfo(data.y, title)
    ShowDfShape(data.y, title)
    ShowDfStats(data.y, title)
    ShowDfNanValues(data.y, title)
    ShowDfHead(data.y, title)
    ShowDfTail(data.y, title)


# Muestra la información del Dataset Split
def ShowDatasetSplitInfo(split: DatasetSplit, title: str, headQty: int = 5):
    tAux = title
    title = f"{tAux} - TRAIN"
    ShowDatasetInfo(split.Train, title)
    title = f"{tAux} - TEST"
    ShowDatasetInfo(split.Test, title)


# Realiza el split del Dataset, en Train y test utilizando el ratio.
def SplitDataset(
    data: Dataset,
    trainRatio: float = TRAIN_RATIO,
    testRatio: float = TEST_RATIO,
    randomState: int = RANDOM_STATE
) -> DatasetSplit:
    if round(trainRatio + testRatio, 10) != 1:
        raise ValueError("trainRatio y testRatio deben sumar 1.")

    XTrain, XTest, yTrain, yTest = train_test_split(
        data.X,
        data.y,
        train_size=trainRatio,
        test_size=testRatio,
        random_state=randomState,
    )
    return DatasetSplit(
        Train=Dataset(X=XTrain.reset_index(drop=True), y=yTrain.reset_index(drop=True)),
        Test=Dataset(X=XTest.reset_index(drop=True), y=yTest.reset_index(drop=True)),
    )


def CreateTrainTestSplit(
    dfClean: pandas.DataFrame,
    targetColumn: str = TARGET_COLUMN,
    trainRatio: float = TRAIN_RATIO,
    testRatio: float = TEST_RATIO,
    stageTitle: str = "Stage 7 - Train Test Split",
) -> DatasetSplit:
    ShowTitleBox(
        stageTitle,
        boxLineStyle=TitleBoxLineStyle.DOUBLE,
        color=ConsoleColor.GREEN,
    )

    if targetColumn not in dfClean.columns:
        raise ValueError(f'No existe la columna objetivo "{targetColumn}" en dfClean.')

    X = DropColumns(dfClean, [targetColumn])
    y = dfClean[[targetColumn]]
    split = SplitDataset(
        Dataset(X=X, y=y),
        trainRatio=trainRatio,
        testRatio=testRatio,
    )

    ShowSuccessMessage(
        f"Split aplicado: "
        f"Train {trainRatio:.0%} - X: {split.Train.X.shape[0]} filas x {split.Train.X.shape[1]} columnas, "
        f"y: {split.Train.y.shape[0]} filas x {split.Train.y.shape[1]} columnas. "
        f"Test {testRatio:.0%} - X: {split.Test.X.shape[0]} filas x {split.Test.X.shape[1]} columnas, "
        f"y: {split.Test.y.shape[0]} filas x {split.Test.y.shape[1]} columnas."
    )
    ShowDatasetSplitHead(split, "dfClean", headQty=10)
    return split


def ShowTrainCorrelationMatrix(
    split: DatasetSplit,
    threshold: float = CORRELATION_THRESHOLD,
    stageTitle: str = "Stage 8 - Train Correlation Matrices",
    resultsDir: Path = RUN_RESULTS_DIR,
    showPlot: bool = SHOW_PLOTS,
    savePlot: bool = SAVE_PLOTS,
):
    ShowTitleBox(
        stageTitle,
        boxLineStyle=TitleBoxLineStyle.DOUBLE,
        color=ConsoleColor.GREEN,
    )
    for level in [
        CorrelationType.ALL,
        CorrelationType.STRONG,
        CorrelationType.WEAK,
    ]:
        ShowDfCorrelation(
            split.Train.X,
            f"X Train - {level.name}",
            level=level,
            umbral=threshold,
            showTable=True,
            figsize=(10, 8),
            outputPath=resultsDir / f"CorrelationMatrix_{level.name}.png",
            showPlot=showPlot,
            savePlot=savePlot,
        )


# Contrato para los escaladores
class ScalerProtocol(Protocol):
    def fit(self, X, y: Any = None) -> Any: ...
    def transform(self, X) -> Any: ...
    def fit_transform(self, X, y: Any = None) -> Any: ...


# Para almacenar los datos del dataset aplicado el escalador.
@dataclass
class ScaledDatasetSplit(DatasetSplit):
    Scaler: ScalerProtocol

# Enum para los tipos de escaladores soportados
class ScalerType(Enum):
    STANDARD = "Standard"
    MIN_MAX = "minmax"
    ROBUST = "robust"
    MAX_ABS = "maxabs"
    NORMALIZER = "normalizer"
    QUANTILE = "quantile"
    POWER = "power"
    FUNCTION = "function"


# Crea una instancia de scaler según el Enum ScalerType.
def CreateScaler(scalerType: ScalerType, **kwargs) -> ScalerProtocol:
    if scalerType == ScalerType.STANDARD:
        return StandardScaler(**kwargs)
    if scalerType == ScalerType.MIN_MAX:
        return MinMaxScaler(**kwargs)
    if scalerType == ScalerType.ROBUST:
        return RobustScaler(**kwargs)
    if scalerType == ScalerType.MAX_ABS:
        return MaxAbsScaler(**kwargs)
    if scalerType == ScalerType.NORMALIZER:
        return Normalizer(**kwargs)
    if scalerType == ScalerType.QUANTILE:
        return QuantileTransformer(**kwargs)
    if scalerType == ScalerType.POWER:
        return PowerTransformer(**kwargs)
    if scalerType == ScalerType.FUNCTION:
        return FunctionTransformer(**kwargs)
    raise ValueError(f"ScalerType no soportado: {scalerType}")

def DetectScaler(scaler: ScalerProtocol) -> ScalerType:
    if isinstance(scaler, StandardScaler):
        return ScalerType.STANDARD
    if isinstance(scaler, MinMaxScaler):
        return ScalerType.MIN_MAX
    if isinstance(scaler, RobustScaler):
        return ScalerType.ROBUST
    if isinstance(scaler, MaxAbsScaler):
        return ScalerType.MAX_ABS
    if isinstance(scaler, Normalizer):
        return ScalerType.NORMALIZER
    if isinstance(scaler, QuantileTransformer):
        return ScalerType.QUANTILE
    if isinstance(scaler, PowerTransformer):
        return ScalerType.POWER
    if isinstance(scaler, FunctionTransformer):
        return ScalerType.FUNCTION
    raise ValueError(f"No se reconoce el tipo de scaler: {type(scaler)}")

# Escala el split usando el escalador proporcionado y retorna el split escalado.
def ScaleDatasetSplit(
    split: DatasetSplit, scaler: ScalerProtocol = StandardScaler()
) -> ScaledDatasetSplit:
    XTrainScaledValues = scaler.fit_transform(split.Train.X)
    XTestScaledValues = scaler.transform(split.Test.X)

    XTrainScaled = pandas.DataFrame(
        XTrainScaledValues, columns=split.Train.X.columns, index=split.Train.X.index
    )

    XTestScaled = pandas.DataFrame(
        XTestScaledValues, columns=split.Test.X.columns, index=split.Test.X.index
    )

    TrainScaledDataset = Dataset(X=XTrainScaled, y=split.Train.y.copy())
    TestScaledDataset = Dataset(X=XTestScaled, y=split.Test.y.copy())

    return ScaledDatasetSplit(
        Train=TrainScaledDataset, Test=TestScaledDataset, Scaler=scaler
    )

# Para almacenar los datos del dataset aplicado PCA.
@dataclass
class PcaDatasetSplit(DatasetSplit):
    Pca: PCA
    Scaler: ScalerProtocol | None = None 

# Aplica PCA al split escalado y retorna el split con PCA aplicado.
def ApplyPCA(
    split: ScaledDatasetSplit,
    explainedVarianceRatioSum: float = 0.95,
    randomState: int = RANDOM_STATE
) -> PcaDatasetSplit:

    def GetPCNames(n: int) -> list[str]:
        return [f"PC{i}" for i in range(1, n + 1)]

    pca = PCA(n_components=explainedVarianceRatioSum, random_state=randomState)

    XTrainPCA = pca.fit_transform(split.Train.X)
    XTestPCA = pca.transform(split.Test.X)

    XTrainPcaDf = pandas.DataFrame(
        XTrainPCA, index=split.Train.X.index, columns=GetPCNames(XTrainPCA.shape[1])
    )

    XTestPcaDf = pandas.DataFrame(
        XTestPCA, index=split.Test.X.index, columns=GetPCNames(XTestPCA.shape[1])
    )

    return PcaDatasetSplit(
        Train=Dataset(X=XTrainPcaDf, y=split.Train.y.copy()),
        Test=Dataset(X=XTestPcaDf, y=split.Test.y.copy()),
        Pca=pca,
        Scaler=split.Scaler
    )

# Tipo de split que puede ser escalado o con PCA aplicado.
SplitLike = ScaledDatasetSplit | PcaDatasetSplit


# Utilidades para detección de tipos de split
@dataclass(frozen=True)
class SplitTypeInfo:
    IsPCA: bool
    IsScaled: bool
    IsRaw: bool

# Detecta el tipo de split (PCA, Escalado, Crudo)
def DetectSplitType(split) -> SplitTypeInfo:
    isPca = isinstance(split, PcaDatasetSplit)
    isScaled = isinstance(split, ScaledDatasetSplit)
    isRaw = not isPca and not isScaled

    return SplitTypeInfo(
        IsPCA=isPca,
        IsScaled=isScaled,
        IsRaw=isRaw
    )


# Función para eliminar filas duplicadas de un DataFrame
def RemoveDfDuplicates(df: pandas.DataFrame, inplace: bool = False) -> pandas.DataFrame:
    if inplace:
        df.drop_duplicates(inplace=True)
        return df
    else:
        return df.drop_duplicates()


# Requested models
def CreatePolynomialRegression(degree: int = POLYNOMIAL_DEGREE, **kwargs):
    return make_pipeline(
        PolynomialFeatures(degree=degree, include_bias=False),
        StandardScaler(),
        LinearRegression(**kwargs)
    )


def CreateRidgeRegression(alpha: float = RIDGE_ALPHA, **kwargs):
    return make_pipeline(
        StandardScaler(),
        Ridge(alpha=alpha, **kwargs)
    )


def ConfigureRegressors(
    stageTitle: str = "Stage 9 - Configure Regressors",
) -> dict[str, Any]:
    ShowTitleBox(
        stageTitle,
        boxLineStyle=TitleBoxLineStyle.DOUBLE,
        color=ConsoleColor.GREEN,
    )
    regressors = {
        "PolynomialRegression": CreatePolynomialRegression(degree=POLYNOMIAL_DEGREE),
        "RidgeRegression": CreateRidgeRegression(alpha=RIDGE_ALPHA),
    }
    ShowSuccessMessage(
        f"Regresores configurados: {', '.join(regressors.keys())}."
    )
    return regressors


@dataclass
class RegressionMetrics:
    MAE: float
    MSE: float
    RMSE: float
    R2: float


@dataclass
class TrainedRegressorResult:
    Name: str
    Model: Any
    Metrics: RegressionMetrics
    Predictions: pandas.DataFrame


def EvaluateRegression(yTrue, yPred) -> RegressionMetrics:
    mse = mean_squared_error(yTrue, yPred)
    return RegressionMetrics(
        MAE=mean_absolute_error(yTrue, yPred),
        MSE=mse,
        RMSE=float(np.sqrt(mse)),
        R2=r2_score(yTrue, yPred)
    )


def ShowRegressionMetrics(results: dict[str, TrainedRegressorResult]):
    metricsRows = []
    for name, result in results.items():
        metricsRows.append({
            "Model": name,
            "MAE": result.Metrics.MAE,
            "MSE": result.Metrics.MSE,
            "RMSE": result.Metrics.RMSE,
            "R2": result.Metrics.R2,
        })
    metricsDf = pandas.DataFrame(metricsRows)
    PrintData(metricsDf.round(4))


def CompareRegressorResults(
    results: dict[str, TrainedRegressorResult],
    stageTitle: str = "Stage 11 - Compare Regressors",
):
    ShowTitleBox(
        stageTitle,
        boxLineStyle=TitleBoxLineStyle.DOUBLE,
        color=ConsoleColor.GREEN,
    )

    bestByRmse = min(results.values(), key=lambda result: result.Metrics.RMSE)
    bestByR2 = max(results.values(), key=lambda result: result.Metrics.R2)

    ShowSuccessMessage(
        f"Mejor modelo según RMSE: {bestByRmse.Name} "
        f"(RMSE={bestByRmse.Metrics.RMSE:.4f})."
    )
    ShowSuccessMessage(
        f"Mejor modelo según R2: {bestByR2.Name} "
        f"(R2={bestByR2.Metrics.R2:.4f})."
    )

    if bestByRmse.Name == bestByR2.Name:
        ShowInfoMessage(f"Modelo recomendado: {bestByRmse.Name}.")
    else:
        ShowWarningMessage(
            "RMSE y R2 recomiendan modelos distintos; revisar el objetivo del análisis."
        )


def PlotRegressionResult(
    modelName: str,
    yTrue: np.ndarray,
    yPred: np.ndarray,
    resultsDir: Path = RUN_RESULTS_DIR,
    showPlot: bool = SHOW_PLOTS,
    savePlot: bool = SAVE_PLOTS,
):
    residuals = yTrue - yPred
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.canvas.manager.set_window_title(f"{modelName} - Regression Results")

    sns.scatterplot(x=yTrue, y=yPred, ax=axes[0])
    minValue = min(yTrue.min(), yPred.min())
    maxValue = max(yTrue.max(), yPred.max())
    axes[0].plot([minValue, maxValue], [minValue, maxValue], color="red", linestyle="--")
    axes[0].set_title(f"{modelName} - Actual vs Predicted")
    axes[0].set_xlabel("Actual")
    axes[0].set_ylabel("Predicted")

    sns.scatterplot(x=yPred, y=residuals, ax=axes[1])
    axes[1].axhline(0, color="red", linestyle="--")
    axes[1].set_title(f"{modelName} - Residuals")
    axes[1].set_xlabel("Predicted")
    axes[1].set_ylabel("Residual")

    plt.tight_layout()
    outputPath = resultsDir / f"{modelName}_RegressionResults.png"
    if savePlot:
        outputPath.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(outputPath, dpi=150)
        ShowSuccessMessage(f'Gráfico guardado en "{outputPath.resolve()}".')
    if showPlot:
        plt.show()
    else:
        plt.close(fig)


def TrainAndEvaluateRegressors(
    regressors: dict[str, Any],
    split: DatasetSplit,
    stageTitle: str = "Stage 10 - Train And Evaluate Regressors",
    resultsDir: Path = RUN_RESULTS_DIR,
    showPlot: bool = SHOW_PLOTS,
    savePlot: bool = SAVE_PLOTS,
) -> dict[str, TrainedRegressorResult]:
    ShowTitleBox(
        stageTitle,
        boxLineStyle=TitleBoxLineStyle.DOUBLE,
        color=ConsoleColor.GREEN,
    )

    results: dict[str, TrainedRegressorResult] = {}
    yTest = split.Test.y.iloc[:, 0].to_numpy()

    for name, regressor in regressors.items():
        ShowInfoMessage(f"Entrenando {name}.")
        regressor.fit(split.Train.X, split.Train.y)
        yPred = np.ravel(regressor.predict(split.Test.X))
        metrics = EvaluateRegression(yTest, yPred)
        predictions = pandas.DataFrame({
            "Actual": yTest,
            "Predicted": yPred,
            "Residual": yTest - yPred,
        })
        results[name] = TrainedRegressorResult(
            Name=name,
            Model=regressor,
            Metrics=metrics,
            Predictions=predictions,
        )
        PlotRegressionResult(
            name,
            yTest,
            yPred,
            resultsDir=resultsDir,
            showPlot=showPlot,
            savePlot=savePlot,
        )

    ShowRegressionMetrics(results)
    return results
    

### ███████████████████████████████████████████████ -------------

def main():
    """Entry point for the regression algorithms script."""
    ClearConsole()
    resultsDir = CreateRunResultsDir()
    ShowDatasetUrl()
    UnzipDatasetArchive()
    dfRaw = LoadRawDataset()
    ShowRawDatasetInfo(dfRaw, dfTitle="dfRaw")
    dfClean = CleanDataFrame(dfRaw, dfTitle="dfClean")
    PlotTargetDistribution(dfClean, resultsDir=resultsDir)
    datasetSplit = CreateTrainTestSplit(dfClean)
    ShowTrainCorrelationMatrix(datasetSplit, resultsDir=resultsDir)
    regressors = ConfigureRegressors()
    trainedRegressors = TrainAndEvaluateRegressors(
        regressors,
        datasetSplit,
        resultsDir=resultsDir,
    )
    CompareRegressorResults(trainedRegressors)


if __name__ == "__main__":
    main()
