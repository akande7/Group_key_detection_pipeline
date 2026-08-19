from collections import defaultdict
from ipaddress import IPv4Address, IPv4Network

import pandas as pd
from scapy.all import (
    PcapReader,
    Dot11,
    Dot11Beacon,
    Ether,
    IP,
    ARP,
    EAPOL,
)

RAW_PCAP_FILE = "attack_phased-01.cap"
DECRYPTED_PCAP_FILE = "attack_phased-01-dec.cap"
OUTPUT_FILE = "attack_features.csv"

WINDOW_SIZE = 10
BROADCAST_MAC = "ff:ff:ff:ff:ff:ff"

LOCAL_NETWORK = IPv4Network("192.168.0.0/24")


def classify_ip_destination(address: str) -> str:
    try:
        ip = IPv4Address(address)
    except ValueError:
        return "invalid"

    if ip == IPv4Address("255.255.255.255"):
        return "limited_broadcast"

    if ip == LOCAL_NETWORK.broadcast_address:
        return "directed_broadcast"

    if ip.is_multicast:
        return "multicast"

    if ip.is_unspecified:
        return "unspecified"

    return "unicast"


def is_broadcast_mac(address: str | None) -> bool:
    return bool(address and address.lower() == BROADCAST_MAC)


def is_multicast_mac(address: str | None) -> bool:
    if not address:
        return False

    address = address.lower()

    if address == BROADCAST_MAC:
        return False

    try:
        first_octet = int(address.split(":")[0], 16)
        return bool(first_octet & 1)
    except (ValueError, IndexError):
        return False


def empty_window() -> dict:
    return {
        "total_packet_count": 0,
        "dot11_frame_count": 0,
        "management_frame_count": 0,
        "control_frame_count": 0,
        "data_frame_count": 0,
        "beacon_count": 0,
        "broadcast_mac_count": 0,
        "multicast_mac_count": 0,
        "unicast_mac_count": 0,
        "eapol_count": 0,
        "protected_frame_count": 0,
        "total_packet_bytes": 0,
        "ip_count": 0,
        "arp_count": 0,
        "broadcast_mac_broadcast_ip_count": 0,
        "broadcast_mac_multicast_ip_count": 0,
        "cross_layer_inconsistency_count": 0,
    }


windows = defaultdict(empty_window)
first_timestamp = None

with PcapReader(RAW_PCAP_FILE) as packets:
    for packet in packets:
        timestamp = float(packet.time)

        if first_timestamp is None:
            first_timestamp = timestamp

        relative_time = timestamp - first_timestamp
        window_number = int(relative_time // WINDOW_SIZE)
        window = windows[window_number]

        window["total_packet_count"] += 1
        window["total_packet_bytes"] += len(packet)

        if not packet.haslayer(Dot11):
            continue

        window["dot11_frame_count"] += 1

        dot11 = packet[Dot11]

        if dot11.type == 0:
            window["management_frame_count"] += 1
        elif dot11.type == 1:
            window["control_frame_count"] += 1
        elif dot11.type == 2:
            window["data_frame_count"] += 1

        if packet.haslayer(Dot11Beacon):
            window["beacon_count"] += 1

        destination_mac = dot11.addr1.lower() if dot11.addr1 else None

        if is_broadcast_mac(destination_mac):
            window["broadcast_mac_count"] += 1
        elif is_multicast_mac(destination_mac):
            window["multicast_mac_count"] += 1
        elif destination_mac:
            window["unicast_mac_count"] += 1

        protected = bool(int(dot11.FCfield) & 0x40)
        if protected:
            window["protected_frame_count"] += 1

        if packet.haslayer(EAPOL):
            window["eapol_count"] += 1


        if not protected and dot11.type == 2:
            if packet.haslayer(ARP):
                window["arp_count"] += 1

            if packet.haslayer(IP):
                window["ip_count"] += 1

                ip_destination = packet[IP].dst
                ip_destination_type = classify_ip_destination(ip_destination)

                if is_broadcast_mac(destination_mac):
                    if ip_destination_type in {
                        "limited_broadcast",
                        "directed_broadcast",
                    }:
                        window["broadcast_mac_broadcast_ip_count"] += 1
                    elif ip_destination_type == "multicast":
                        window["broadcast_mac_multicast_ip_count"] += 1
                    elif ip_destination_type == "unicast":
                        window["cross_layer_inconsistency_count"] += 1


decrypted_first_timestamp = None

with PcapReader(DECRYPTED_PCAP_FILE) as packets:
    for packet in packets:
        if not packet.haslayer(Ether):
            continue

        timestamp = float(packet.time)

        if decrypted_first_timestamp is None:
            decrypted_first_timestamp = timestamp

        relative_time = timestamp - decrypted_first_timestamp
        window_number = int(relative_time // WINDOW_SIZE)
        window = windows[window_number]

        destination_mac = packet[Ether].dst.lower()

        if packet.haslayer(ARP):
            window["arp_count"] += 1

        if not packet.haslayer(IP):
            continue

        window["ip_count"] += 1

        ip_destination = packet[IP].dst
        ip_destination_type = classify_ip_destination(ip_destination)

        if is_broadcast_mac(destination_mac):
            if ip_destination_type in {
                "limited_broadcast",
                "directed_broadcast",
            }:
                window["broadcast_mac_broadcast_ip_count"] += 1
            elif ip_destination_type == "multicast":
                window["broadcast_mac_multicast_ip_count"] += 1
            elif ip_destination_type == "unicast":
                window["cross_layer_inconsistency_count"] += 1

rows = []

for window_number in sorted(windows):
    values = windows[window_number]

    total_packets = values["total_packet_count"]
    dot11_frames = values["dot11_frame_count"]

    row = {
        "window_number": window_number,
        "window_start_seconds": window_number * WINDOW_SIZE,
        "window_end_seconds": (window_number + 1) * WINDOW_SIZE,
        **values,
        "average_packet_size": (
            values["total_packet_bytes"] / total_packets
            if total_packets
            else 0
        ),
        "packet_rate_per_second": total_packets / WINDOW_SIZE,
        "broadcast_mac_ratio": (
            values["broadcast_mac_count"] / dot11_frames
            if dot11_frames
            else 0
        ),
        "unicast_mac_ratio": (
            values["unicast_mac_count"] / dot11_frames
            if dot11_frames
            else 0
        ),
        "cross_layer_inconsistency_ratio": (
            values["cross_layer_inconsistency_count"]
            / values["ip_count"]
            if values["ip_count"]
            else 0
        ),
    }

    rows.append(row)


dataframe = pd.DataFrame(rows)
dataframe.to_csv(OUTPUT_FILE, index=False)

print("\nFeature extraction complete (v2 - handles plaintext + decrypted IP)")
print("-" * 50)
print(f"Raw capture: {RAW_PCAP_FILE}")
print(f"Decrypted capture: {DECRYPTED_PCAP_FILE}")
print(f"Number of windows: {len(dataframe)}")
print(f"Output CSV: {OUTPUT_FILE}")
print("\nExtracted features")
print("-" * 50)
print(dataframe.to_string(index=False))