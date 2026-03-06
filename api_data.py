import logging
from typing import Any

from livekit.agents import Agent, RunContext, function_tool

from data_tools import load_csv, df_overview, missing_values, top_rows, value_counts, group_mean

logger = logging.getLogger("csv-voice-analyst")
logger.setLevel(logging.INFO)


class CsvAgent(Agent):
    def __init__(self, instructions: str): #constructor to store instructions
        super().__init__(instructions=instructions)
        self._df_cache: dict[str, Any] = {}  # filename -> DataFrame

    def _df(self, filename: str):
        if filename not in self._df_cache:
            self._df_cache[filename] = load_csv(filename)
        return self._df_cache[filename]

    @function_tool(description="Show basic info about a CSV: number of rows, columns, and column names.")
    async def csv_overview(self, context: RunContext, filename: str) -> dict[str, Any]:
        df = self._df(filename)
        info = df_overview(df)
        return {
            "message": f"{filename} has {info['rows']} rows and {info['cols']} columns. Columns: {', '.join(info['columns'])}"
        }

    @function_tool(description="Show the first N rows of a CSV.")
    async def csv_head(self, context: RunContext, filename: str, n: int = 5) -> dict[str, Any]:
        df = self._df(filename)
        rows = top_rows(df, n=max(1, min(n, 20)))
        return {"rows": rows, "message": f"Here are the first {len(rows)} rows from {filename}."}

    @function_tool(description="Show which columns have missing values.")
    async def csv_missing(self, context: RunContext, filename: str, top_n: int = 10) -> dict[str, Any]:
        df = self._df(filename)
        mv = missing_values(df, top_n=max(1, min(top_n, 50)))
        if not mv["missing"]:
            return {"message": f"No missing values found in {filename}."}
        return {"missing": mv["missing"], "message": f"Top missing-value columns in {filename}."}

    @function_tool(description="Get top value counts for a column in a CSV.")
    async def csv_value_counts(self, context: RunContext, filename: str, column: str, n: int = 10) -> dict[str, Any]:
        df = self._df(filename)
        counts = value_counts(df, column, n=max(1, min(n, 50)))
        return {"counts": counts["counts"], "message": f"Top values in {column} from {filename}."}

    @function_tool(description="Compute mean of a metric column grouped by another column.")
    async def csv_group_mean(self, context: RunContext, filename: str, group_col: str, metric_col: str, n: int = 10) -> dict[str, Any]:
        df = self._df(filename)
        out = group_mean(df, group_col, metric_col, n=max(1, min(n, 50)))
        return {"mean_by_group": out["mean_by_group"], "message": f"Mean {metric_col} by {group_col} (top {n})."}
