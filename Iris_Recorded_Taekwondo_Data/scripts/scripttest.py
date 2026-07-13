import argparse
import json
import os
from typing import Any


def find_sync_manifest(directory: str) -> str | None:
    """Find the sync manifest JSON file in a session directory."""
    candidates = []
    for name in os.listdir(directory):
        path = os.path.join(directory, name)
        if os.path.isfile(path) and name.lower().endswith('.json') and 'sync' in name.lower():
            candidates.append((name.lower(), path))
    if not candidates:
        return None
    candidates.sort()
    for name, path in candidates:
        if name == 'sync_manifest.json':
            return path
    return candidates[0][1]


def count_playback_events(value: Any) -> dict[str, int]:
    """Count playback start and stop events in a parsed JSON structure."""
    counts = {'playback_start': 0, 'playback_stop': 0}

    if isinstance(value, dict):
        events = value.get('events')
        if isinstance(events, list):
            for event in events:
                if not isinstance(event, dict):
                    continue
                event_name = str(event.get('event', '')).strip().lower()
                if event_name.endswith('_playback_start'):
                    counts['playback_start'] += 1
                elif event_name.endswith('_playback_stop'):
                    counts['playback_stop'] += 1

        for key, item in value.items():
            if key == 'events':
                continue
            if isinstance(item, (dict, list)):
                sub_counts = count_playback_events(item)
                counts['playback_start'] += sub_counts['playback_start']
                counts['playback_stop'] += sub_counts['playback_stop']
        return counts

    if isinstance(value, list):
        for item in value:
            sub_counts = count_playback_events(item)
            counts['playback_start'] += sub_counts['playback_start']
            counts['playback_stop'] += sub_counts['playback_stop']

    return counts


def session_playback_summary(root_directory: str) -> dict[str, dict[str, int]]:
    summary: dict[str, dict[str, int]] = {}
    if not os.path.isdir(root_directory):
        raise FileNotFoundError(f'Root directory not found: {root_directory}')

    for entry in sorted(os.listdir(root_directory)):
        session_path = os.path.join(root_directory, entry)
        if not os.path.isdir(session_path):
            continue
        manifest_path = find_sync_manifest(session_path)
        if not manifest_path:
            continue
        try:
            with open(manifest_path, 'r', encoding='utf-8') as handle:
                manifest = json.load(handle)
        except (json.JSONDecodeError, OSError):
            continue
        counts = count_playback_events(manifest)
        summary[entry] = counts
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description='Count playback start and stop events in each session sync manifest.'
    )
    default_root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
    parser.add_argument(
        'root',
        nargs='?',
        default=default_root,
        help='Root directory containing session folders (default: parent of scripts folder).'
    )
    args = parser.parse_args()

    summary = session_playback_summary(args.root)
    if not summary:
        print('No session sync manifest files found.')
        return

    for session, counts in sorted(summary.items()):
        print(f'{session}: starts={counts["playback_start"]}, stops={counts["playback_stop"]}')


if __name__ == '__main__':
    main()
