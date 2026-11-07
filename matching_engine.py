import numpy as np

class HybridMatcher:
    def __init__(self):
        # Initialize default weights: [w1(Text), w2(MBTI), w3(Location)]
        # We start with neutral weights, which the ML layer will later optimize
        self.weights = np.array([0.5, 0.3, 0.2])

    def get_mbti_score(self, mbti1, mbti2):
        # Perfect match
        if mbti1 == mbti2:
            return 1.0
        # Shared Introvert/Extrovert baseline
        if mbti1[0] == mbti2[0]:
            return 0.8
        # Mixed energy pairing
        return 0.5

    def get_location_score(self, loc1, loc2):
        # 1.0 for same city, 0.0 for different cities
        return 1.0 if loc1 == loc2 else 0.0

    def calculate_compatibility(self, text_sim, mbti_score, loc_score):
        # Combine the three component scores into a feature array
        features = np.array([text_sim, mbti_score, loc_score])
        
        # Calculate the dot product (w1*f1 + w2*f2 + w3*f3) and scale to 100%
        raw_score = np.dot(self.weights, features) * 100
        
        # Ensure the final output is strictly bounded between 0 and 100
        return max(0.0, min(100.0, raw_score))

if __name__ == "__main__":
    matcher = HybridMatcher()
    
    # Test Case: Two introverts (INTJ and INTP) in the same city (Bangalore)
    # Let's assume the NLP engine gave their bios a 0.4242 similarity score
    text_score = 0.4242
    mbti_score = matcher.get_mbti_score("INTJ", "INTP")
    loc_score = matcher.get_location_score("Bangalore", "Bangalore")
    
    final_match = matcher.calculate_compatibility(text_score, mbti_score, loc_score)
    
    print(f"Component Scores -> Text: {text_score:.2f} | MBTI: {mbti_score:.2f} | Loc: {loc_score:.2f}")
    print(f"Final Compatibility Score: {final_match:.2f}%")