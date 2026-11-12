"""
Canonical station catalog for all 18 Indian synoptic monitoring nodes.
Shared across data_pipeline.py, db.py, and dashboard.py so station metadata
remains a single source of truth.
"""

from typing import Any, Dict, List

INDIAN_MET_STATIONS: List[Dict[str, Any]] = [
    {"name": "New Delhi (Safdarjung)", "state": "Delhi", "lat": 28.5847, "lon": 77.2066, "base_temp": 31.0, "zone": "North Plains", "is_coastal": False},
    {"name": "Mumbai (Santacruz)", "state": "Maharashtra", "lat": 19.0760, "lon": 72.8777, "base_temp": 32.5, "zone": "West Coast", "is_coastal": True},
    {"name": "Kolkata (Alipore)", "state": "West Bengal", "lat": 22.5726, "lon": 88.3639, "base_temp": 33.0, "zone": "East Coast / BoB", "is_coastal": True},
    {"name": "Chennai (Meenambakkam)", "state": "Tamil Nadu", "lat": 13.0827, "lon": 80.2707, "base_temp": 34.0, "zone": "South Coast", "is_coastal": True},
    {"name": "Bengaluru (HAL)", "state": "Karnataka", "lat": 12.9716, "lon": 77.5946, "base_temp": 27.5, "zone": "Deccan Plateau", "is_coastal": False},
    {"name": "Hyderabad (Begumpet)", "state": "Telangana", "lat": 17.3850, "lon": 78.4867, "base_temp": 32.0, "zone": "Central Peninsula", "is_coastal": False},
    {"name": "Bhubaneswar (Airport)", "state": "Odisha", "lat": 20.2961, "lon": 85.8245, "base_temp": 33.5, "zone": "East Coast / BoB", "is_coastal": True},
    {"name": "Srinagar (Rambagh)", "state": "J&K", "lat": 34.0837, "lon": 74.7973, "base_temp": 18.0, "zone": "Western Himalaya", "is_coastal": False},
    {"name": "Guwahati (Borjhar)", "state": "Assam", "lat": 26.1445, "lon": 91.7362, "base_temp": 29.5, "zone": "Northeast Basin", "is_coastal": False},
    {"name": "Ahmedabad (Airport)", "state": "Gujarat", "lat": 23.0225, "lon": 72.5714, "base_temp": 35.0, "zone": "West Arid", "is_coastal": False},
    {"name": "Jaipur (Sanganer)", "state": "Rajasthan", "lat": 26.9124, "lon": 75.7873, "base_temp": 34.5, "zone": "Northwest Semi-Arid", "is_coastal": False},
    {"name": "Nagpur (Sonegaon)", "state": "Maharashtra", "lat": 21.1458, "lon": 79.0882, "base_temp": 33.0, "zone": "Central India", "is_coastal": False},
    {"name": "Visakhapatnam (CWC)", "state": "Andhra Pradesh", "lat": 17.6868, "lon": 83.2185, "base_temp": 32.8, "zone": "East Coast / BoB", "is_coastal": True},
    {"name": "Kochi (CIAL)", "state": "Kerala", "lat": 9.9312, "lon": 76.2673, "base_temp": 30.5, "zone": "Southwest Coast", "is_coastal": True},
    {"name": "Patna (Aerodrome)", "state": "Bihar", "lat": 25.5941, "lon": 85.1376, "base_temp": 32.0, "zone": "Gangetic Plain", "is_coastal": False},
    {"name": "Bhopal (Bairagarh)", "state": "Madhya Pradesh", "lat": 23.2599, "lon": 77.4126, "base_temp": 31.5, "zone": "Central India", "is_coastal": False},
    {"name": "Pune (Shivajinagar)", "state": "Maharashtra", "lat": 18.5204, "lon": 73.8567, "base_temp": 30.0, "zone": "Western Ghats", "is_coastal": False},
    {"name": "Dehradun (FRI)", "state": "Uttarakhand", "lat": 30.3165, "lon": 78.0322, "base_temp": 24.5, "zone": "Sub-Himalayan", "is_coastal": False},
]
