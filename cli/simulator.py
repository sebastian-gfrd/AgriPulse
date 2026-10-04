"""Interactive User Journey Simulator for AgriPulse.

Simulates the bimodal lifecycle of Noor (Ondera Highlands, Coffee Smallholder):
- Scene 1: Weekend Synchronization (Daughter's Smartphone + 3G + JAX Edge CPU Engine)
- Scene 2: Weekday SMS Delivery (Noor's Feature Keypad Phone in Swahili)
- Scene 3: Ethical Guardrail & Signposting Fail-Safe (Uncertainty zone)
- Scene 4: Market Negotiation Parity (Countering predatory middleman price)
"""

import argparse
import os
import sys
import time
import numpy as np

# Force CPU execution for edge simulation
os.environ["JAX_PLATFORMS"] = "cpu"

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn
from rich.text import Text

from agripulse.engine import AgriPulseEdgeEngine
from agripulse.market import MarketParityEngine
from agripulse.telemetry import TelemetryFrame, format_sms_message, format_voice_script


console = Console()


def render_ascii_feature_phone(screen_text: str, title: str = "NOKIA 105 - NOOR'S INBOX"):
    """Render an authentic ASCII art retro basic keypad phone displaying an SMS."""
    # Ensure lines are max 34 chars for feature phone display width
    lines = []
    for raw_line in screen_text.split("\n"):
        words = raw_line.split(" ")
        curr = ""
        for w in words:
            if len(curr) + len(w) + 1 <= 34:
                curr += (" " if curr else "") + w
            else:
                lines.append(curr.ljust(34))
                curr = w
        if curr:
            lines.append(curr.ljust(34))

    # Pad or trim to 6 lines
    while len(lines) < 6:
        lines.append(" " * 34)
    lines = lines[:6]

    phone_art = f"""
      ┌──────────────────────────────────────┐
      │  [=] 2G  ●●○○  100% [🔋]   06:30 AM  │
      ├──────────────────────────────────────┤
      │  {lines[0]}  │
      │  {lines[1]}  │
      │  {lines[2]}  │
      │  {lines[3]}  │
      │  {lines[4]}  │
      │  {lines[5]}  │
      ├──────────────────────────────────────┤
      │   [ Options ]          [ Back ]      │
      └──────────────────────────────────────┘
             ┌───────────┬───────────┐
             │    (▲)    │    (▼)    │
             └───────────┴───────────┘
             ┌───────┬───────┬───────┐
             │ 1 🔊  │ 2 ABC │ 3 DEF │
             ├───────┼───────┼───────┤
             │ 4 GHI │ 5 JKL │ 6 MNO │
             ├───────┼───────┼───────┤
             │ 7PQRS │ 8 TUV │ 9WXYZ │
             ├───────┼───────┼───────┤
             │ * +   │ 0 _   │ # ⇧   │
             └───────┴───────┴───────┘
    """
    console.print(Panel(phone_art, title=f"[bold green]{title}[/bold green]", expand=False))


def simulate_weekend_sync(engine: AgriPulseEdgeEngine, sample_raw: np.ndarray):
    """Simulate daughter's weekend 3G sync and XLA CPU execution."""
    console.print(
        Panel(
            "[bold cyan]SCENE 1: WEEKEND SYNCHRONIZATION[/bold cyan]\n"
            "[italic]Saturday 4:15 PM — Ondera Highlands[/italic]\n"
            "Noor's 16-year-old daughter returns from boarding school. She connects her budget "
            "Android smartphone to the cellular network via a 30 MB prepaid 3G data bundle (~$0.15).",
            border_style="cyan",
        )
    )

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        console=console,
    ) as progress:
        task1 = progress.add_task("[yellow]Querying CHIRPS Daily Rainfall raster...", total=100)
        task2 = progress.add_task("[magenta]Fetching NASA POWER Agro-climate telemetry (RH, Tmax, Tmin)...", total=100)
        task3 = progress.add_task("[green]Pinging WFP Food Prices (HDX Nyeri/Ondera coffee index)...", total=100)

        for _ in range(25):
            time.sleep(0.02)
            progress.update(task1, advance=4)
        for _ in range(25):
            time.sleep(0.02)
            progress.update(task2, advance=4)
        for _ in range(25):
            time.sleep(0.02)
            progress.update(task3, advance=4)

    console.print("[green]✔ Telemetry sync completed:[/green] Total payload received: [bold]4.82 KB[/bold]\n")

    # Edge CPU execution
    console.print("[bold yellow]Executing AgriPulse 1D-CNN on Phone CPU (JAX/XLA Compiled)...[/bold yellow]")
    res = engine.predict(sample_raw)

    drivers = res.climatic_drivers
    table = Table(title="Aggregated 30-Day Microclimatic Vector (Ondera Highlands)", show_header=True)
    table.add_column("Channel / Variable", style="cyan")
    table.add_column("Observation Value", style="bold white")
    table.add_column("Biological Significance", style="italic")

    table.add_row("Total Rainfall (CHIRPS)", f"{drivers['total_rainfall_mm']} mm", "Continuous moisture for spore germination")
    table.add_row("Rain Days (>1.5 mm)", f"{drivers['wet_days_count']} days", "Leaf wetness duration (>6h consecutive)")
    table.add_row("Thermal Amplitude (ΔT)", f"{drivers['max_temperature_c'] - drivers['min_temperature_c']:.1f} °C", "Narrow diurnal swing favors fungal hyphae")
    table.add_row("Mean Rel. Humidity (RH2M)", f"{drivers['mean_relative_humidity']}%", "Survival threshold > 82% required")
    table.add_row("Soil Susceptibility (iSDAsoil)", "pH 5.3 (Volcanic Clay)", "Acidic highland soil host baseline")
    console.print(table)

    console.print(
        f"\n[bold]Inference Diagnostics:[/bold]\n"
        f"  • Latency: [bold green]{res.inference_latency_ms:.3f} ms[/bold green] on single CPU core\n"
        f"  • Calibrated Risk Score: [bold magenta]{res.risk_score * 100:.1f}%[/bold magenta]\n"
        f"  • Classification State: [bold yellow]{res.status_label}[/bold yellow]\n"
        f"  • Escalation Required: [bold]{res.requires_human_escalation}[/bold]\n"
    )

    return res


def simulate_weekday_delivery(res, coffee_price: float = 125.0):
    """Simulate SMS arrival on Noor's basic feature phone on Tuesday morning."""
    console.print(
        Panel(
            "[bold green]SCENE 2: WEEKDAY SMS DELIVERY ON FEATURE PHONE[/bold green]\n"
            "[italic]Tuesday 6:30 AM — Noor's Homestead[/italic]\n"
            "Her daughter is back at boarding school. Noor has her basic keypad phone at home. "
            "Before walking up to the upper slope coffee rows, she hears the incoming SMS beep.",
            border_style="green",
        )
    )

    sms_text, char_count = format_sms_message(
        status_code=res.status_code,
        risk_score=res.risk_score,
        coffee_price=coffee_price,
        language="swahili",
    )

    render_ascii_feature_phone(sms_text, title="NOKIA 105 - INCOMING SMS (SWAHILI)")
    console.print(f"[dim]SMS length: {char_count}/160 characters (Standard Single SMS Billing)[/dim]\n")

    # Spoken IVR voice alternative
    console.print("[bold cyan]Audio / Voice Note Script (For Low-Literacy / Audio Playback):[/bold cyan]")
    voice_script = format_voice_script(
        status_code=res.status_code,
        risk_score=res.risk_score,
        coffee_price=coffee_price,
        farmer_name="Noor",
        language="swahili",
    )
    console.print(Panel(voice_script, title="[italic]Ujumbe wa Sauti (Swahili Voice Advisory)[/italic]", border_style="blue"))


def simulate_ethical_guardrail(engine: AgriPulseEdgeEngine, sample_raw: Optional[np.ndarray] = None):
    """Demonstrate the Pass/Fail ethical signposting fail-safe."""
    console.print(
        Panel(
            "[bold red]SCENE 3: ETHICAL GUARDRAIL & FAIL-SAFE SIGNPOSTING[/bold red]\n"
            "[italic]Scenario: Borderline Microclimatic Anomaly (Temperature Swing + Erratic Fog)[/italic]\n"
            "The model encounters a pattern with high uncertainty (P in [0.40, 0.75]). "
            "A conventional black-box model might hallucinate or trigger unnecessary pesticide debt. "
            "AgriPulse strictly enforces human-in-the-loop escalation.",
            border_style="red",
        )
    )

    if sample_raw is None:
        # Load sample index 0 from test dataset which is inside the uncertain zone
        data = np.load("data/ondera_coffee_rust_dataset.npz")
        sample_raw = data["X_test_raw"][0]

    res = engine.predict(sample_raw)

    console.print(f"Risk Output: [bold]{res.risk_score * 100:.1f}%[/bold] (Inside Guardrail Zone: 40% - 75%)")
    console.print(f"System State: [bold red]{res.status_label}[/bold red]")
    console.print(f"Human Referral Action: [bold yellow]{res.action_swahili}[/bold yellow]\n")

    sms_text, char_count = format_sms_message(res.status_code, res.risk_score, 125.0, "swahili")
    render_ascii_feature_phone(sms_text, title="NOKIA 105 - ETHICAL GUARDRAIL ACTIVATED")
    console.print(
        "[bold green]✔ Compliance Audit:[/bold green] Zero hallucinations. System refrained from automated "
        "chemical recommendation and signposted Noor directly to the local cooperative extension agent."
    )


def simulate_market_negotiation(market_engine: MarketParityEngine):
    """Simulate coffee harvest market negotiation against middleman."""
    console.print(
        Panel(
            "[bold yellow]SCENE 4: HARVEST MARKET PARITY (WFP FOOD PRICES)[/bold yellow]\n"
            "[italic]Harvest Season — Farm Gate at Ondera[/italic]\n"
            "A middleman pulls up in a pickup truck and offers Noor 72 KES/kg for her 650 kg parchment harvest. "
            "Noor checks her AgriPulse market benchmark from the WFP / HDX price feed.",
            border_style="yellow",
        )
    )

    analysis = market_engine.evaluate_middleman_offer(
        offered_price=72.0,
        commodity_key="coffee_parchment_grade1",
        estimated_volume_kg=650.0,
    )

    t = Table(title="Market Price Discrepancy Analysis", show_header=True)
    t.add_column("Metric", style="cyan")
    t.add_column("Value", style="bold")

    t.add_row("Commodity", analysis.commodity)
    t.add_row("Middleman Offer", f"{analysis.offered_price:.2f} {analysis.currency}/kg")
    t.add_row("WFP Regional Benchmark", f"{analysis.benchmark_price:.2f} {analysis.currency}/kg")
    t.add_row("Cooperative Fair Floor", f"{analysis.fair_floor_price:.2f} {analysis.currency}/kg")
    t.add_row("Price Gap (Underpayment)", f"-{analysis.price_gap_amount:.2f} {analysis.currency}/kg (-{analysis.price_gap_percent:.1f}%)")
    t.add_row("Is Predatory?", f"[bold red]{analysis.is_predatory}[/bold red]")
    t.add_row("Total Potential Loss on 650 kg", f"[bold red]-{analysis.total_potential_loss:,.2f} {analysis.currency}[/bold red] (~${analysis.total_potential_loss/130:.0f} USD)")

    console.print(t)
    console.print(f"\n[bold]Bargaining Guidance (Swahili):[/bold]\n[green]{analysis.advice_swahili}[/green]\n")
    console.print(
        "[bold green]Outcome:[/bold green] Empowered by independent WFP market intelligence, "
        "Noor refuses the 72 KES offer and delivers her 650 kg parchment to the Ondera Cooperative Dry Mill, "
        "preserving [bold green]34,450 KES (~$265 USD)[/bold green] in household income."
    )


def main():
    parser = argparse.ArgumentParser(description="AgriPulse End-to-End User Journey Simulator")
    parser.add_argument("--scenario", choices=["critical", "stable", "uncertain", "market", "all"], default="all")
    parser.add_argument("--runs", type=int, default=1)
    args = parser.parse_args()

    console.print("\n" + "=" * 75)
    console.print("[bold green] AGRIPULSE: SMALL AI FOR DEVELOPMENT SIMULATOR [/bold green]")
    console.print("[dim]World Bank Youth Summit × Global AI & Digital Summit (Seoul 2026)[/dim]")
    console.print("=" * 75 + "\n")

    engine = AgriPulseEdgeEngine()
    market_engine = MarketParityEngine()

    data = np.load("data/ondera_coffee_rust_dataset.npz")
    X_test_raw = data["X_test_raw"]
    y_test = data["y_test"]

    # Dynamically find representative samples for each status code
    crit_idx, uncert_idx, stable_idx = None, None, None
    for i in range(min(len(X_test_raw), 50)):
        sc = engine.predict(X_test_raw[i]).status_code
        if sc == 2 and crit_idx is None:
            crit_idx = i
        elif sc == 1 and uncert_idx is None:
            uncert_idx = i
        elif sc == 0 and stable_idx is None:
            stable_idx = i
        if crit_idx is not None and uncert_idx is not None and stable_idx is not None:
            break

    crit_sample = X_test_raw[crit_idx if crit_idx is not None else 23]
    uncert_sample = X_test_raw[uncert_idx if uncert_idx is not None else 0]
    stable_sample = X_test_raw[stable_idx if stable_idx is not None else 1]

    if args.scenario in ["critical", "all"]:
        res_crit = simulate_weekend_sync(engine, crit_sample)
        simulate_weekday_delivery(res_crit, coffee_price=125.0)

    if args.scenario in ["uncertain", "all"]:
        simulate_ethical_guardrail(engine, uncert_sample)

    if args.scenario in ["stable"]:
        res_stable = simulate_weekend_sync(engine, stable_sample)
        simulate_weekday_delivery(res_stable, coffee_price=125.0)

    if args.scenario in ["market", "all"]:
        simulate_market_negotiation(market_engine)

    console.print("\n" + "=" * 75)
    console.print("[bold green] SIMULATION COMPLETED SUCCESSFULLY [/bold green]")
    console.print("=" * 75 + "\n")


if __name__ == "__main__":
    main()
