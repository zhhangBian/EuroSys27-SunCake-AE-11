"""Map the existing application metadata to the TokenCake HTTP protocol."""

from uuid import uuid4


def request_metadata(agent_info: dict) -> dict:
    fields = {
        "type": "agent_type",
        "name": "agent_name",
        "priority": "importance",
        "app_start_time": "application_started_at_s",
        "app_start_offset": "application_start_offset_s",
        "app_elapsed_time": "application_elapsed_s",
        "app_max_depth": "application_max_depth",
    }
    metadata = {
        fields.get(key, key): value
        for key, value in agent_info.items()
        if key not in {"application_id", "start_time", "resume_deadline"}
    }
    metadata["lifecycle_id"] = "tc-" + uuid4().hex
    return metadata
