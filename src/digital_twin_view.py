"""
Virtual CNC Machine Visualization Module for Digital Twin (PRJ_422).
Generates dynamic, real-time SVG and UI components representing the physical CNC
milling machine, its active kinematic state, spindle, cutter wear, and coolant systems.
"""

from typing import Dict, Any
from utils.config import (
    STATUS_NORMAL,
    STATUS_WARNING,
    STATUS_CRITICAL,
    WEAR_WARNING_UM,
    WEAR_CRITICAL_UM,
    WEAR_MAX_THRESHOLD_UM
)


def generate_cnc_machine_svg(
    cut_id: int,
    spindle_speed_rpm: float,
    feed_rate_mm_min: float,
    depth_of_cut_mm: float,
    coolant_mode: str,
    flank_wear_um: float,
    tool_health_pct: float,
    machine_status: str,
    force_res_n: float,
    vibration_rms_g: float
) -> str:
    """
    Generate responsive, high-fidelity SVG graphic of the CNC Milling Machine.
    Dynamically responds to machine condition (Normal / Warning / Critical),
    tool wear stage, coolant delivery, and cutter kinematics.
    """
    # 1. State colors and status parameters
    status_upper = machine_status.upper()
    if status_upper == STATUS_CRITICAL:
        theme_color = "#ef4444"
        tool_color = "#ef4444"
        tool_glow = "rgba(239, 68, 68, 0.8)"
        andon_red = "#ef4444"
        andon_amber = "rgba(245, 158, 11, 0.2)"
        andon_green = "rgba(16, 185, 129, 0.2)"
        chamber_glow = "rgba(239, 68, 68, 0.12)"
        op_state_label = "CRITICAL ALARM // EMERGENCY RETRACT"
        sparks_display = "inline"
    elif status_upper == STATUS_WARNING or "WARN" in status_upper:
        theme_color = "#f59e0b"
        tool_color = "#f59e0b"
        tool_glow = "rgba(245, 158, 11, 0.7)"
        andon_red = "rgba(239, 68, 68, 0.2)"
        andon_amber = "#f59e0b"
        andon_green = "rgba(16, 185, 129, 0.2)"
        chamber_glow = "rgba(245, 158, 11, 0.08)"
        op_state_label = "WARNING ADVISORY // FEED-HOLD RECOMMENDED"
        sparks_display = "inline"
    else:
        theme_color = "#10b981"
        tool_color = "#38bdf8"
        tool_glow = "rgba(16, 185, 129, 0.5)"
        andon_red = "rgba(239, 68, 68, 0.2)"
        andon_amber = "rgba(245, 158, 11, 0.2)"
        andon_green = "#10b981"
        chamber_glow = "rgba(56, 189, 248, 0.05)"
        op_state_label = "G01 ACTIVE LINEAR MILLING CYCLE"
        sparks_display = "none"

    # 2. Tool tip kinematics
    # Table X travel: tool moves horizontally across stock
    # Stock sits in center between x=300 and x=460 (width 160)
    x_offset = ((cut_id % 40) / 40.0) * 110.0 - 55.0  # [-55, +55]
    cutter_center_x = 380 + x_offset
    cutter_tip_y = 310 + min(20.0, depth_of_cut_mm * 8.0)  # Engaged in workpiece

    # Coolant spray styling
    coolant_lower = coolant_mode.lower()
    if "dry" in coolant_lower:
        coolant_disp = "none"
        coolant_text = "DRY / AIR PURGE"
    elif "mist" in coolant_lower:
        coolant_disp = "inline"
        coolant_opacity = "0.35"
        coolant_color = "#93c5fd"
        coolant_text = "MIST (MQL) ACTIVE"
    else:
        coolant_disp = "inline"
        coolant_opacity = "0.75"
        coolant_color = "#38bdf8"
        coolant_text = "FLOOD COOLANT ACTIVE"

    # Estimated cutting zone temperature proxy (°C)
    temp_c = int(180 + flank_wear_um * 3.8 + (force_res_n / 4.0))

    svg = f"""
    <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 760 510" width="100%" height="100%" style="background:#090d16; border-radius:12px; font-family:'JetBrains Mono', monospace;">
        <defs>
            <!-- Linear Gradients for Metal Casting & Shading -->
            <linearGradient id="gantryGrad" x1="0" y1="0" x2="1" y2="0">
                <stop offset="0%" stop-color="#1e293b"/>
                <stop offset="50%" stop-color="#334155"/>
                <stop offset="100%" stop-color="#1e293b"/>
            </linearGradient>
            <linearGradient id="spindleBodyGrad" x1="0" y1="0" x2="1" y2="0">
                <stop offset="0%" stop-color="#0f172a"/>
                <stop offset="40%" stop-color="#475569"/>
                <stop offset="70%" stop-color="#334155"/>
                <stop offset="100%" stop-color="#0f172a"/>
            </linearGradient>
            <linearGradient id="chuckGrad" x1="0" y1="0" x2="1" y2="0">
                <stop offset="0%" stop-color="#475569"/>
                <stop offset="50%" stop-color="#94a3b8"/>
                <stop offset="100%" stop-color="#334155"/>
            </linearGradient>
            <linearGradient id="tableGrad" x1="0" y1="0" x2="0" y2="1">
                <stop offset="0%" stop-color="#334155"/>
                <stop offset="100%" stop-color="#1e293b"/>
            </linearGradient>
            <linearGradient id="stockGrad" x1="0" y1="0" x2="1" y2="0">
                <stop offset="0%" stop-color="#64748b"/>
                <stop offset="50%" stop-color="#cbd5e1"/>
                <stop offset="100%" stop-color="#475569"/>
            </linearGradient>
            <linearGradient id="toolGrad" x1="0" y1="0" x2="0" y2="1">
                <stop offset="0%" stop-color="#cbd5e1"/>
                <stop offset="70%" stop-color="{tool_color}"/>
                <stop offset="100%" stop-color="{tool_color}"/>
            </linearGradient>
            <!-- Coolant Spray Pattern -->
            <linearGradient id="sprayGradL" x1="0" y1="0" x2="1" y2="1">
                <stop offset="0%" stop-color="#38bdf8" stop-opacity="0.9"/>
                <stop offset="100%" stop-color="#0284c7" stop-opacity="0.1"/>
            </linearGradient>
            <linearGradient id="sprayGradR" x1="1" y1="0" x2="0" y2="1">
                <stop offset="0%" stop-color="#38bdf8" stop-opacity="0.9"/>
                <stop offset="100%" stop-color="#0284c7" stop-opacity="0.1"/>
            </linearGradient>
            <!-- Filters -->
            <filter id="glowFilter" x="-20%" y="-20%" width="140%" height="140%">
                <feGaussianBlur stdDeviation="4" result="blur"/>
                <feMerge>
                    <feMergeNode in="blur"/>
                    <feMergeNode in="SourceGraphic"/>
                </feMerge>
            </filter>
        </defs>

        <!-- ================= 1. CNC ENCLOSURE & CHAMBER ================= -->
        <!-- Outer Enclosure Shell -->
        <rect x="15" y="25" width="730" height="470" rx="14" fill="#0f172a" stroke="#334155" stroke-width="2"/>
        <!-- Top Roof Gantry Trim -->
        <path d="M 15 65 L 745 65" stroke="#1e293b" stroke-width="3"/>
        <rect x="15" y="25" width="730" height="40" rx="14" fill="#1e293b"/>
        <text x="35" y="50" fill="#94a3b8" font-size="12" font-weight="700" letter-spacing="1">HAAS VF-2SS // 5-AXIS VMC VIRTUAL TWIN</text>
        <text x="540" y="50" fill="{theme_color}" font-size="11" font-weight="700">● {status_upper}</text>

        <!-- Top Andon Tower Stack Light -->
        <rect x="690" y="8" width="18" height="38" rx="4" fill="#0b0f19" stroke="#475569" stroke-width="1.5"/>
        <rect x="697" y="46" width="4" height="20" fill="#64748b"/>
        <!-- Red Lamp -->
        <circle cx="699" cy="16" r="4.5" fill="{andon_red}" filter="{('url(#glowFilter)' if status_upper == STATUS_CRITICAL else 'none')}"/>
        <!-- Amber Lamp -->
        <circle cx="699" cy="27" r="4.5" fill="{andon_amber}" filter="{('url(#glowFilter)' if 'WARN' in status_upper else 'none')}"/>
        <!-- Green Lamp -->
        <circle cx="699" cy="38" r="4.5" fill="{andon_green}" filter="{('url(#glowFilter)' if status_upper == STATUS_NORMAL else 'none')}"/>

        <!-- Machining Chamber Window Area -->
        <rect x="35" y="80" width="690" height="360" rx="8" fill="#090d16" stroke="#1e293b" stroke-width="2"/>
        <rect x="35" y="80" width="690" height="360" rx="8" fill="{chamber_glow}"/>

        <!-- Accordion Way-Covers (Bellows) on Back Wall -->
        <g stroke="#1e293b" stroke-width="2" fill="none">
            <path d="M 50 110 L 710 110 M 50 135 L 710 135 M 50 160 L 710 160 M 50 185 L 710 185 M 50 210 L 710 210 M 50 235 L 710 235"/>
        </g>

        <!-- Chamber Grid Lines -->
        <g stroke="rgba(255,255,255,0.03)" stroke-width="1">
            <line x1="380" y1="80" x2="380" y2="440"/>
            <line x1="35" y1="310" x2="725" y2="310"/>
        </g>

        <!-- ================= 2. MACHINE BED & WORKPIECE ================= -->
        <!-- Cast Iron Machine Table (X-Y Axis Bed) -->
        <rect x="180" y="370" width="400" height="48" rx="4" fill="url(#tableGrad)" stroke="#475569" stroke-width="1.5"/>
        <!-- T-Slots in Table -->
        <rect x="200" y="382" width="360" height="6" fill="#0f172a"/>
        <rect x="200" y="398" width="360" height="6" fill="#0f172a"/>

        <!-- Kurt Precision Machine Vise -->
        <rect x="250" y="335" width="260" height="35" rx="3" fill="#334155" stroke="#64748b" stroke-width="1.5"/>
        <!-- Vise Hardened Jaws -->
        <rect x="270" y="315" width="22" height="25" fill="#475569" stroke="#94a3b8" stroke-width="1"/>
        <rect x="468" y="315" width="22" height="25" fill="#475569" stroke="#94a3b8" stroke-width="1"/>

        <!-- Workpiece Billet (Inconel 718) -->
        <rect x="292" y="310" width="176" height="30" rx="2" fill="url(#stockGrad)" stroke="#94a3b8" stroke-width="1.5"/>
        <text x="325" y="328" fill="#1e293b" font-size="10" font-weight="800">INCONEL 718 (AMS 5662)</text>

        <!-- Milled Slot / Depth of Cut Cavity (dynamically rendered) -->
        <rect x="310" y="310" width="140" height="{min(18.0, max(4.0, depth_of_cut_mm * 8.0))}" fill="#1e293b" stroke="#38bdf8" stroke-width="1" stroke-dasharray="2 2"/>

        <!-- ================= 3. SPINDLE HEAD & CUTTER (Z-AXIS) ================= -->
        <!-- Z-Axis Ram / Spindle Housing -->
        <rect x="{cutter_center_x - 48}" y="80" width="96" height="115" rx="5" fill="url(#spindleBodyGrad)" stroke="#475569" stroke-width="2"/>
        <!-- Spindle Cooling Fins -->
        <g stroke="#64748b" stroke-width="1">
            <line x1="{cutter_center_x - 42}" y1="100" x2="{cutter_center_x + 42}" y2="100"/>
            <line x1="{cutter_center_x - 42}" y1="115" x2="{cutter_center_x + 42}" y2="115"/>
            <line x1="{cutter_center_x - 42}" y1="130" x2="{cutter_center_x + 42}" y2="130"/>
            <line x1="{cutter_center_x - 42}" y1="145" x2="{cutter_center_x + 42}" y2="145"/>
            <line x1="{cutter_center_x - 42}" y1="160" x2="{cutter_center_x + 42}" y2="160"/>
        </g>
        <!-- Spindle Motor Logo/Badge -->
        <rect x="{cutter_center_x - 30}" y="115" width="60" height="24" rx="3" fill="#0f172a" stroke="#3b82f6" stroke-width="1"/>
        <text x="{cutter_center_x - 22}" y="131" fill="#38bdf8" font-size="9" font-weight="700">10,400 RPM</text>

        <!-- Spindle Rotating Cartridge Taper -->
        <path d="M {cutter_center_x - 35} 195 L {cutter_center_x + 35} 195 L {cutter_center_x + 24} 230 L {cutter_center_x - 24} 230 Z" fill="url(#chuckGrad)" stroke="#64748b" stroke-width="1.5"/>

        <!-- BT40 Collet Chuck Body -->
        <rect x="{cutter_center_x - 20}" y="230" width="40" height="28" rx="2" fill="#1e293b" stroke="#64748b" stroke-width="1.5"/>
        <rect x="{cutter_center_x - 14}" y="258" width="28" height="12" fill="#334155" stroke="#94a3b8" stroke-width="1"/>

        <!-- 4-Flute Tungsten Carbide End-Mill (Ø6mm) -->
        <!-- Shank -->
        <rect x="{cutter_center_x - 5}" y="270" width="10" height="20" fill="#94a3b8" stroke="#475569" stroke-width="0.8"/>
        <!-- Fluted Cutting Section (Glows based on wear) -->
        <rect x="{cutter_center_x - 5}" y="290" width="10" height="{cutter_tip_y - 290}" fill="url(#toolGrad)" stroke="{tool_color}" stroke-width="1.5" filter="url(#glowFilter)"/>
        <!-- Helical Flute Texture Lines -->
        <g stroke="#0f172a" stroke-width="1.2">
            <line x1="{cutter_center_x - 5}" y1="294" x2="{cutter_center_x + 5}" y2="299"/>
            <line x1="{cutter_center_x - 5}" y1="302" x2="{cutter_center_x + 5}" y2="307"/>
            <line x1="{cutter_center_x - 5}" y1="310" x2="{cutter_center_x + 5}" y2="315"/>
        </g>
        <!-- Tool Wear / Thermal Halo Indicator at cutting tip -->
        <circle cx="{cutter_center_x}" cy="{cutter_tip_y}" r="9" fill="{tool_glow}" opacity="0.65" filter="url(#glowFilter)"/>

        <!-- ================= 4. COOLANT SYSTEM ================= -->
        <!-- Left Loc-Line Articulated Nozzle -->
        <g stroke="#0284c7" stroke-width="5" stroke-linecap="round" fill="none">
            <path d="M {cutter_center_x - 48} 175 Q {cutter_center_x - 70} 220 {cutter_center_x - 24} 280"/>
        </g>
        <!-- Right Loc-Line Articulated Nozzle -->
        <g stroke="#0284c7" stroke-width="5" stroke-linecap="round" fill="none">
            <path d="M {cutter_center_x + 48} 175 Q {cutter_center_x + 70} 220 {cutter_center_x + 24} 280"/>
        </g>

        <!-- Coolant Spray Plumes (Dual High-Pressure Jet) -->
        <g display="{coolant_disp}">
            <polygon points="{cutter_center_x - 24},280 {cutter_center_x + 2},312 {cutter_center_x - 12},318" fill="url(#sprayGradL)" opacity="{coolant_opacity}"/>
            <polygon points="{cutter_center_x + 24},280 {cutter_center_x - 2},312 {cutter_center_x + 12},318" fill="url(#sprayGradR)" opacity="{coolant_opacity}"/>
            <ellipse cx="{cutter_center_x}" cy="{cutter_tip_y + 3}" rx="14" ry="4" fill="#38bdf8" opacity="0.4"/>
        </g>

        <!-- Dynamic Friction Sparks (Visible on Warning/Critical Wear) -->
        <g display="{sparks_display}">
            <line x1="{cutter_center_x - 2}" y1="{cutter_tip_y}" x2="{cutter_center_x - 18}" y2="{cutter_tip_y - 12}" stroke="#f59e0b" stroke-width="1.8" filter="url(#glowFilter)"/>
            <line x1="{cutter_center_x + 2}" y1="{cutter_tip_y}" x2="{cutter_center_x + 16}" y2="{cutter_tip_y - 16}" stroke="#ef4444" stroke-width="1.8" filter="url(#glowFilter)"/>
            <line x1="{cutter_center_x}" y1="{cutter_tip_y}" x2="{cutter_center_x + 22}" y2="{cutter_tip_y - 6}" stroke="#fbbf24" stroke-width="1.5" filter="url(#glowFilter)"/>
            <line x1="{cutter_center_x - 3}" y1="{cutter_tip_y}" x2="{cutter_center_x - 14}" y2="{cutter_tip_y + 8}" stroke="#f97316" stroke-width="1.5"/>
        </g>

        <!-- ================= 5. HUD OVERLAY & MACHINE LABELS ================= -->
        <!-- Kinematic Crosshair Tracker -->
        <g stroke="{theme_color}" stroke-width="0.8" opacity="0.65" stroke-dasharray="3 3">
            <line x1="{cutter_center_x - 30}" y1="{cutter_tip_y}" x2="{cutter_center_x + 30}" y2="{cutter_tip_y}"/>
            <line x1="{cutter_center_x}" y1="{cutter_tip_y - 30}" x2="{cutter_center_x}" y2="{cutter_tip_y + 30}"/>
            <circle cx="{cutter_center_x}" cy="{cutter_tip_y}" r="15" fill="none"/>
        </g>

        <!-- Top Right Chamber Telemetry Overlay -->
        <rect x="490" y="90" width="225" height="100" rx="6" fill="rgba(15, 23, 42, 0.85)" stroke="#334155" stroke-width="1"/>
        <text x="502" y="110" fill="#94a3b8" font-size="10" font-weight="700">KINEMATICS & PROCESS HUD</text>
        <text x="502" y="130" fill="#f8fafc" font-size="11">X: <tspan fill="#38bdf8">{x_offset:+.1f} mm</tspan> | Y: <tspan fill="#38bdf8">-12.4 mm</tspan></text>
        <text x="502" y="148" fill="#f8fafc" font-size="11">Z DEPTH: <tspan fill="#ec4899">-{depth_of_cut_mm:.2f} mm</tspan></text>
        <text x="502" y="166" fill="#f8fafc" font-size="11">FEED: <tspan fill="#10b981">{feed_rate_mm_min:.0f} mm/min</tspan></text>
        <text x="502" y="182" fill="#f8fafc" font-size="10">ZONE TEMP: <tspan fill="{theme_color}">~{temp_c} °C</tspan></text>

        <!-- Bottom Operating Banner -->
        <rect x="35" y="445" width="690" height="40" rx="6" fill="#131b2e" stroke="#334155" stroke-width="1.2"/>
        <circle cx="55" cy="465" r="5" fill="{theme_color}" filter="url(#glowFilter)"/>
        <text x="70" y="469" fill="#f8fafc" font-size="11" font-weight="700">{op_state_label}</text>
        <text x="490" y="469" fill="#94a3b8" font-size="10">COOLANT: <tspan fill="#38bdf8">{coolant_text}</tspan></text>
    </svg>
    """
    return svg
