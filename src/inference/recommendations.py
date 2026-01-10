"""Treatment recommendations engine."""


# Hardcoded treatment recommendations for apple diseases
APPLE_DISEASE_RECOMMENDATIONS = {
    "Healthy": {
        "treatment": "No treatment needed. Continue regular maintenance.",
        "prevention": "Maintain good orchard hygiene. Monitor leaves regularly for signs of disease."
    },
    "Scab": {
        "treatment": "Apply fungicide sprays (e.g., sulfur or copper-based). Remove infected leaves.",
        "prevention": "Prune for good air circulation. Remove fallen leaves. Apply preventive fungicides in spring."
    },
    "Black Rot": {
        "treatment": "Remove infected fruit and branches. Apply fungicide. Prune affected areas.",
        "prevention": "Maintain good sanitation. Remove mummified fruit. Apply preventive fungicides."
    },
    "Cedar Rust": {
        "treatment": "Apply fungicide sprays. Remove infected leaves and fruit.",
        "prevention": "Remove nearby cedar/juniper trees if possible. Apply preventive fungicides. Prune for air circulation."
    }
}


def get_recommendations(disease_name):
    """
    Get treatment recommendations for a disease.
    
    Args:
        disease_name (str): Name of the disease
        
    Returns:
        tuple: (treatment, prevention) strings, or (None, None) if disease not found
    """
    rec = APPLE_DISEASE_RECOMMENDATIONS.get(disease_name, None)
    if rec is None:
        return None, None
    return rec['treatment'], rec['prevention']


def get_all_recommendations():
    """
    Get all available recommendations.
    
    Returns:
        dict: All disease recommendations
    """
    return APPLE_DISEASE_RECOMMENDATIONS.copy()


def format_recommendations(disease_name):
    """
    Format recommendations as a readable string.
    
    Args:
        disease_name (str): Name of the disease
        
    Returns:
        str: Formatted recommendations or error message
    """
    recommendations = get_recommendations(disease_name)
    
    if recommendations is None:
        return f"No recommendations available for {disease_name}. Please consult an agronomist."
    
    return (
        f"Disease: {disease_name}\n\n"
        f"Treatment:\n{recommendations['treatment']}\n\n"
        f"Prevention:\n{recommendations['prevention']}\n\n"
        f"Note: For severe cases, consult an agronomist."
    )
