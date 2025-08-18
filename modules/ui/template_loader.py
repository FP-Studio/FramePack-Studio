"""
Template loader utility for UI components.
Uses Python's built-in string.Template for safe variable substitution.
"""

import os
from string import Template
from typing import Dict, Optional


class TemplateLoader:
    """Loads and renders templates using Python's built-in Template class."""

    def __init__(self, template_dir: str = None):
        """Initialize template loader with template directory."""
        if template_dir is None:
            # Default to templates directory next to this file
            current_dir = os.path.dirname(__file__)
            template_dir = os.path.join(current_dir, "templates")

        self.template_dir = template_dir
        self._template_cache: Dict[str, Template] = {}

    def load_template(self, template_name: str) -> Template:
        """Load template from file, with caching."""
        if template_name not in self._template_cache:
            template_path = os.path.join(self.template_dir, template_name)

            if not os.path.exists(template_path):
                raise FileNotFoundError(f"Template not found: {template_path}")

            with open(template_path, "r", encoding="utf-8") as f:
                template_content = f.read()

            self._template_cache[template_name] = Template(template_content)

        return self._template_cache[template_name]

    def render_template(self, template_name: str, **kwargs) -> str:
        """Render template with provided variables."""
        template = self.load_template(template_name)
        return template.safe_substitute(**kwargs)

    def render_markdown_template(self, template_name: str, **kwargs) -> str:
        """Render markdown template with provided variables."""
        return self.render_template(template_name, **kwargs)

    def render_html_template(self, template_name: str, **kwargs) -> str:
        """Render HTML template with provided variables."""
        return self.render_template(template_name, **kwargs)

    def load_css(self, css_name: str) -> str:
        """Load CSS file content directly (no templating)."""
        css_path = os.path.join(self.template_dir, css_name)

        if not os.path.exists(css_path):
            raise FileNotFoundError(f"CSS file not found: {css_path}")

        with open(css_path, "r", encoding="utf-8") as f:
            return f.read()


# Global template loader instance
template_loader = TemplateLoader()


def render_thumbnail_html(thumbnail_url: Optional[str]) -> str:
    """Render thumbnail HTML with safe template substitution."""
    if not thumbnail_url:
        return ""

    return template_loader.render_html_template(
        "thumbnail.html", thumbnail_url=thumbnail_url
    )


def get_queue_documentation() -> str:
    """Get queue documentation markdown content."""
    return template_loader.render_markdown_template("queue_documentation.md")


def get_status_icon(status: str) -> str:
    """Get the SVG icon for a status."""
    status = status.lower()

    if status == "running":
        # Return the custom running animation
        return """
        <rect x="0" y="0" width="2" height="2" fill="lightgrey">
            <animate attributeName="fill" values="darkgray;lightgrey" begin="0ms" dur="960ms" repeatCount="indefinite" />
        </rect>
        <rect x="4" y="0" width="2" height="2" fill="lightgrey">
            <animate attributeName="fill" values="darkgray;lightgrey" begin="150ms" dur="960ms" repeatCount="indefinite" />
        </rect>
        <rect x="4" y="4" width="2" height="2" fill="lightgrey">
            <animate attributeName="fill" values="darkgray;lightgrey" begin="300ms" dur="960ms" repeatCount="indefinite" />
        </rect>
        <rect x="4" y="8" width="2" height="2" fill="lightgrey">
            <animate attributeName="fill" values="darkgray;lightgrey" begin="450ms" dur="960ms" repeatCount="indefinite" />
        </rect>
        <rect x="0" y="8" width="2" height="2" fill="lightgrey">
            <animate attributeName="fill" values="darkgray;lightgrey" begin="600ms" dur="960ms" repeatCount="indefinite" />
        </rect>
        <rect x="0" y="4" width="2" height="2" fill="lightgrey">
            <animate attributeName="fill" values="darkgray;lightgrey" begin="750ms" dur="960ms" repeatCount="indefinite" />
        </rect>
        """

    if status == "cancelling":
        # Return animated cancelling icon (X in circle with pulsing animation)
        return """
        <circle cx="12" cy="12" r="10" fill="none" stroke="currentColor" stroke-width="2" opacity="0.8">
            <animate attributeName="opacity" values="0.8;0.4;0.8" dur="1200ms" repeatCount="indefinite" />
        </circle>
        <path d="M15 9L9 15M9 9L15 15" stroke="currentColor" stroke-width="2" stroke-linecap="round">
            <animate attributeName="opacity" values="0.6;1;0.6" dur="1200ms" repeatCount="indefinite" />
        </path>
        """

    # Static icons using paths
    icons = {
        "pending": "M12 2C6.5 2 2 6.5 2 12C2 17.5 6.5 22 12 22C17.5 22 22 17.5 22 12C22 6.5 17.5 2 12 2M12 20C7.59 20 4 16.41 4 12C4 7.59 7.59 4 12 4C16.41 4 20 7.59 20 12C20 16.41 16.41 20 12 20M15 12L12 12V7H10V14H15V12Z",
        "completed": "M9,20.42L2.79,14.21L5.62,11.38L9,14.77L18.88,4.88L21.71,7.71L9,20.42Z",
        "failed": "M13,13H11V7H13M13,17H11V15H13M12,2A10,10 0 0,0 2,12A10,10 0 0,0 12,22A10,10 0 0,0 22,12A10,10 0 0,0 12,2Z",
        "cancelled": "M12,2L1,21H23M12,6L19.53,19H4.47M11,10V13H13V10M11,15V17H13V15",
    }

    if status in icons:
        return f'<path d="{icons[status]}"/>'

    return f'<path d="{icons["pending"]}"/>'


def render_setting_pill(label: str, value: str) -> str:
    """Render a single setting pill."""
    return template_loader.render_html_template(
        "setting_pill.html", label=label, value=value
    )


def get_changed_settings_pills(job_data: dict) -> str:
    """Generate pills for settings that differ from defaults."""
    # Define comprehensive default values based on actual usage patterns
    default_settings = {
        "prompt": "",
        "negative_prompt": "",
        "steps": 25,
        "cfg": 1.0,
        "gs": 10,
        "rs": 0,
        "latent_type": "Noise",
        "latent_window_size": 9,
        "resolutionW": 640,
        "resolutionH": 640,
        "model_type": "Original",
        "generation_type": "Original",
        "total_second_length": 6,
        "blend_sections": 4,
        "num_cleaned_frames": 0,
        "end_frame_strength": 1.0,
        "end_frame_image_path": None,
        "end_frame_used": "False",
        "input_video": None,
        "video_path": None,
        "x_param": None,
        "y_param": None,
        "x_values": None,
        "y_values": None,
        "combine_with_source": True,
        "use_teacache": False,
        "teacache_num_steps": 25,
        "teacache_rel_l1_thresh": 0.15,
        "use_magcache": True,
        "magcache_threshold": 0.1,
        "magcache_max_consecutive_skips": 2,
        "magcache_retention_ratio": 0.25,
        "loras": {},
    }

    # Settings to always show (these will be handled separately in the template)
    always_show = {
        "model_type",
        "generation_type",
        "resolutionW",
        "resolutionH",
        "seed",
        "prompt",
    }

    # Settings that should not be shown as pills (internal/system fields)
    never_show = {
        "prompt",
        "negative_prompt",
        "seed",
        "end_frame_image_path",
        "input_video",
        "video_path",
        "x_param",
        "y_param",
        "x_values",
        "y_values",
    }

    # Custom label mappings for better display
    label_mappings = {
        "cfg": "CFG",
        "gs": "Guidance Scale",
        "rs": "Rescale",
        "latent_type": "Latent Type",
        "latent_window_size": "Window Size",
        "resolutionW": "Width",
        "resolutionH": "Height",
        "total_second_length": "Duration",
        "blend_sections": "Blend Sections",
        "num_cleaned_frames": "Cleaned Frames",
        "end_frame_strength": "End Frame Strength",
        "end_frame_used": "End Frame",
        "combine_with_source": "Combine Source",
        "use_teacache": "TeaCache",
        "teacache_num_steps": "TeaCache Steps",
        "teacache_rel_l1_thresh": "TeaCache Threshold",
        "use_magcache": "MagCache",
        "magcache_threshold": "MagCache Threshold",
        "magcache_max_consecutive_skips": "MagCache Max Skips",
        "magcache_retention_ratio": "MagCache Retention",
    }

    pills = []
    for key, value in job_data.items():
        if key not in always_show and key not in never_show:
            default_value = default_settings.get(key)

            # Skip if value matches default (handle None values properly)
            if value == default_value or (value is None and default_value is None):
                continue

            # Skip empty loras dict
            if key == "loras" and (not value or value == {}):
                continue

            # Format the value based on type
            if isinstance(value, float):
                formatted_value = (
                    f"{value:.2f}" if value != int(value) else str(int(value))
                )
            elif isinstance(value, bool):
                formatted_value = "Yes" if value else "No"
            elif value is None:
                formatted_value = "None"
            elif key == "loras" and isinstance(value, dict):
                # Show LoRA count if any are loaded
                lora_count = len([k for k, v in value.items() if v != 0])
                if lora_count > 0:
                    formatted_value = (
                        f"{lora_count} LoRA{'s' if lora_count != 1 else ''}"
                    )
                else:
                    continue
            else:
                formatted_value = str(value)

            # Get custom label or generate from key
            label = label_mappings.get(key, key.replace("_", " ").title())

            pills.append(render_setting_pill(label, formatted_value))

    return "\n".join(pills)


def render_queue_row(
    job_id: str,
    job_id_full: str,
    generation_type: str,
    status: str,
    prompt_text: str,
    size: str,
    length: str,
    seed: str,
    started: str,
    completed: str,
    elapsed: str,
    thumbnail: str,
    job_data: dict,
) -> str:
    """Render a single queue row."""
    status_class = status.lower()
    status_icon = get_status_icon(status)
    settings_pills = get_changed_settings_pills(job_data)

    # Set viewBox based on status
    viewbox = "0 0 6 10" if status_class == "running" else "0 0 24 24"

    return template_loader.render_html_template(
        "queue_row.html",
        job_id=job_id,
        job_id_full=job_id_full,
        generation_type=generation_type,
        status_class=status_class,
        status_icon=status_icon,
        viewbox=viewbox,
        prompt_text=prompt_text,
        size=size,
        length=length,
        seed=seed,
        started=started,
        completed=completed,
        elapsed=elapsed,
        thumbnail=thumbnail,
        settings_pills=settings_pills,
    )


def render_queue(rows: list) -> str:
    """Render the complete queue with all rows."""
    queue_rows = "\n".join(rows)
    return template_loader.render_html_template("queue.html", queue_rows=queue_rows)
