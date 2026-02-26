"""Domain-specific scenario generators for manufacturing, surveillance, and traffic.

Provides pre-configured scene templates and prompt builders for each supported
domain, enabling rapid synthetic dataset generation without manual scene design.
"""

from __future__ import annotations

import random
from typing import Any

from aumos_common.observability import get_logger

from aumos_video_engine.core.models import VideoDomain

logger = get_logger(__name__)


class ManufacturingScenarioGenerator:
    """Generates manufacturing QA scenario configurations.

    Creates assembly line, robotic arm, and conveyor belt scene configurations
    for training visual inspection and defect detection models.
    """

    ASSEMBLY_LINE_TEMPLATES: list[dict[str, Any]] = [
        {
            "name": "assembly-line-standard",
            "description": "Standard assembly line with conveyor belt and overhead lighting",
            "scene_config": {
                "lighting": [
                    {"type": "AREA", "position": [0, 0, 4], "energy": 2000},
                    {"type": "AREA", "position": [-3, 0, 4], "energy": 1000},
                    {"type": "AREA", "position": [3, 0, 4], "energy": 1000},
                ],
                "camera_path": [
                    {"position": [0, -3, 2], "forward": [0, 1, -0.5]},
                    {"position": [0.5, -3, 2], "forward": [-0.2, 1, -0.5]},
                ],
                "environment": {"type": "solid_color", "color": [0.15, 0.15, 0.15]},
                "render_settings": {"samples": 128, "noise_threshold": 0.005},
            },
        },
        {
            "name": "robotic-arm-inspection",
            "description": "Robotic arm with part inspection station",
            "scene_config": {
                "lighting": [
                    {"type": "SPOT", "position": [0, 0, 3], "energy": 3000, "angle": 30},
                ],
                "camera_path": [
                    {"position": [2, -2, 1.5], "forward": [-0.7, 0.7, -0.5]},
                ],
                "environment": {"type": "solid_color", "color": [0.1, 0.1, 0.1]},
                "render_settings": {"samples": 256, "noise_threshold": 0.002},
            },
        },
    ]

    def generate_prompt(
        self,
        defect_rate: float = 0.0,
        lighting: str = "standard",
        camera_angle: str = "overhead",
        product_type: str = "electronic_component",
    ) -> str:
        """Generate a manufacturing QA text prompt.

        Args:
            defect_rate: Probability (0.0–1.0) of defects appearing in the scene.
            lighting: Lighting condition: "standard", "dim", "harsh".
            camera_angle: Camera angle: "overhead", "side", "angled".
            product_type: Type of product being inspected.

        Returns:
            Text prompt for video generation.
        """
        defect_desc = ""
        if defect_rate > 0.3:
            defect_desc = "with visible surface defects and scratches "
        elif defect_rate > 0.0:
            defect_desc = "with minor blemishes "

        lighting_desc = {"standard": "under bright industrial lighting", "dim": "in low light conditions", "harsh": "under harsh direct lighting"}.get(lighting, "")

        angle_desc = {"overhead": "from overhead camera", "side": "from side-mounted camera", "angled": "from angled inspection camera"}.get(camera_angle, "")

        return (
            f"Industrial assembly line conveyor belt with {product_type} parts "
            f"moving {defect_desc}{lighting_desc} {angle_desc}, "
            f"manufacturing quality assurance inspection, synthetic training data"
        )

    def get_default_scene_config(self, variant: str = "assembly-line-standard") -> dict[str, Any]:
        """Get default scene configuration for a manufacturing template variant.

        Args:
            variant: Template variant name.

        Returns:
            Scene configuration dict.
        """
        for template in self.ASSEMBLY_LINE_TEMPLATES:
            if template["name"] == variant:
                return dict(template["scene_config"])
        return dict(self.ASSEMBLY_LINE_TEMPLATES[0]["scene_config"])


class SurveillanceScenarioGenerator:
    """Generates surveillance and security camera scenario configurations.

    Creates indoor and outdoor security footage configurations for training
    crowd monitoring, anomaly detection, and intrusion detection models.
    All generated footage is suitable for privacy-enforced training datasets.
    """

    SCENARIO_TYPES = ["lobby", "parking_lot", "corridor", "street", "retail_floor"]

    def generate_prompt(
        self,
        scenario: str = "lobby",
        crowd_density: str = "low",
        time_of_day: str = "day",
        anomaly: str | None = None,
    ) -> str:
        """Generate a surveillance scenario text prompt.

        Args:
            scenario: Location type: lobby, parking_lot, corridor, street, retail_floor.
            crowd_density: "empty", "low", "medium", "high".
            time_of_day: "day", "night", "dawn_dusk".
            anomaly: Optional anomaly to include: "loitering", "package_left", "intrusion".

        Returns:
            Text prompt for video generation.
        """
        density_desc = {
            "empty": "empty",
            "low": "a few people walking",
            "medium": "moderate foot traffic",
            "high": "crowded with many people",
        }.get(crowd_density, "")

        time_desc = {
            "day": "in daylight",
            "night": "at night with artificial lighting",
            "dawn_dusk": "in low ambient light",
        }.get(time_of_day, "")

        anomaly_desc = ""
        if anomaly == "loitering":
            anomaly_desc = "with a person loitering suspiciously "
        elif anomaly == "package_left":
            anomaly_desc = "with an unattended package left on the floor "
        elif anomaly == "intrusion":
            anomaly_desc = "showing unauthorized access to restricted area "

        return (
            f"Security camera footage of {scenario} with {density_desc} "
            f"{anomaly_desc}{time_desc}, "
            f"CCTV surveillance perspective, synthetic training data for anomaly detection, "
            f"all faces privacy-protected"
        )

    def get_scene_config(
        self,
        scenario: str = "lobby",
        time_of_day: str = "day",
    ) -> dict[str, Any]:
        """Get scene configuration for a surveillance scenario.

        Args:
            scenario: Location type.
            time_of_day: Lighting time condition.

        Returns:
            Scene configuration dict.
        """
        ambient_energy = 500 if time_of_day == "day" else 50
        return {
            "lighting": [
                {"type": "SUN" if time_of_day == "day" else "POINT", "position": [5, 5, 10], "energy": ambient_energy},
            ],
            "camera_path": [
                {"position": [0, 0, 3], "forward": [0, 0, -1]},  # Top-down CCTV mount
            ],
            "environment": {
                "type": "sky" if time_of_day == "day" else "solid_color",
                "color": [0.05, 0.05, 0.05] if time_of_day == "night" else None,
            },
            "render_settings": {"samples": 64},
        }


class TrafficScenarioGenerator:
    """Generates traffic and autonomous vehicle scenario configurations.

    Creates intersection, highway, and urban driving footage for training
    autonomous vehicle perception models with configurable weather and
    rare edge case scenarios.
    """

    WEATHER_CONDITIONS = ["clear", "overcast", "rain", "fog", "snow"]
    SCENARIO_TYPES = ["intersection", "highway", "urban_street", "parking"]

    def generate_prompt(
        self,
        scenario: str = "intersection",
        vehicle_density: str = "medium",
        weather: str = "clear",
        rare_event: str | None = None,
        time_of_day: str = "day",
    ) -> str:
        """Generate a traffic/AV scenario text prompt.

        Args:
            scenario: Traffic scenario type.
            vehicle_density: "sparse", "medium", "heavy".
            weather: Weather condition.
            rare_event: Optional rare event: "pedestrian_jaywalking", "debris",
                "emergency_vehicle", "cyclist_swerve".
            time_of_day: "day", "night", "dawn_dusk".

        Returns:
            Text prompt for video generation.
        """
        density_desc = {
            "sparse": "light traffic",
            "medium": "moderate traffic",
            "heavy": "heavy congested traffic",
        }.get(vehicle_density, "")

        weather_desc = {
            "clear": "in clear weather",
            "overcast": "under overcast skies",
            "rain": "in rain with wet roads",
            "fog": "in dense fog with reduced visibility",
            "snow": "in snow with icy roads",
        }.get(weather, "")

        rare_event_desc = ""
        if rare_event == "pedestrian_jaywalking":
            rare_event_desc = "with pedestrian crossing unexpectedly "
        elif rare_event == "debris":
            rare_event_desc = "with road debris obstruction "
        elif rare_event == "emergency_vehicle":
            rare_event_desc = "with emergency vehicle with sirens "
        elif rare_event == "cyclist_swerve":
            rare_event_desc = "with cyclist swerving into lane "

        return (
            f"Dashcam perspective of {scenario} with {density_desc} "
            f"{rare_event_desc}{weather_desc} at {time_of_day}, "
            f"autonomous vehicle training dataset, "
            f"all license plates privacy-redacted"
        )

    def generate_rare_event_configs(self, num_scenarios: int = 10) -> list[dict[str, Any]]:
        """Generate a randomized set of rare event scenario configurations.

        Useful for creating balanced rare-event datasets for edge case training.

        Args:
            num_scenarios: Number of rare event scenarios to generate.

        Returns:
            List of scenario configuration dicts.
        """
        rare_events = ["pedestrian_jaywalking", "debris", "emergency_vehicle", "cyclist_swerve"]
        scenarios = []
        for _ in range(num_scenarios):
            scenario = random.choice(self.SCENARIO_TYPES)
            weather = random.choice(self.WEATHER_CONDITIONS)
            event = random.choice(rare_events)
            prompt = self.generate_prompt(
                scenario=scenario,
                vehicle_density=random.choice(["sparse", "medium", "heavy"]),
                weather=weather,
                rare_event=event,
                time_of_day=random.choice(["day", "night"]),
            )
            scenarios.append({
                "scenario": scenario,
                "weather": weather,
                "rare_event": event,
                "prompt": prompt,
                "domain": "traffic",
            })
        return scenarios
