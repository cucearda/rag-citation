"""
Example usage of the citation module for extracting claims and retrieving evidence.

This script demonstrates how to:
1. Extract claims from a paragraph of text
2. Query Pinecone for supporting evidence
3. Display the results with citations
"""

from citation import extract_claims_with_citations
from pinecone_store import PineconeStore


def main():
    """Run citation extraction example."""
    
    print("=" * 80)
    print("CITATION EXTRACTION DEMO")
    print("=" * 80)
    
    # Example paragraph that needs citations
    paragraph = """
    Hope theory has significant implications for entrepreneurship research and practice.
    Entrepreneurs face uncertainty. They must maintain goals despite setbacks.
    Research shows that hopeful individuals are more likely to persist 
    in challenging business environments and adapt their strategies effectively.
    Moreover, hope has been linked to better performance outcomes.
    The pathway thinking component helps entrepreneurs identify alternative routes.
    Agency thinking provides the motivation to pursue goals.
    """
    
    print("\n📝 INPUT PARAGRAPH:")
    print("-" * 80)
    print(paragraph.strip())
    print()
    
    # Initialize Pinecone store
    print("🔌 Connecting to Pinecone...")
    store = PineconeStore(index_name="rag-citation-index")
    print("✓ Connected\n")
    
    # Extract claims with citations
    print("🔍 Extracting claims and retrieving evidence (top_k=15)...")
    claims = extract_claims_with_citations(
        paragraph=paragraph,
        store=store,
        namespace="default",
        top_k=15
    )
    print(f"✓ Found {len(claims)} claims\n")
    
    # Display results
    print("=" * 80)
    print("EXTRACTED CLAIMS WITH EVIDENCE")
    print("=" * 80)
    
    for claim in claims:
        print(f"\n📌 CLAIM {claim.index}:")
        print(f"   Text: {claim.text}")
        print(f"   Word count: {claim.word_count}")
        print(f"   Sentence indices: {claim.sentence_indices}")
        print(f"   Evidence sources found: {len(claim.evidence)}")
        
        if claim.evidence:
            print(f"\n   Top 3 Evidence Sources:")
            for i, evidence in enumerate(claim.evidence[:3], 1):
                print(f"\n   {i}. Score: {evidence.score:.4f}")
                print(f"      Title: {evidence.title}")
                print(f"      Section: {evidence.section_title}")
                print(f"      Pages: {', '.join(str(p) for p in evidence.pages)}")
                print(f"      Text preview: {evidence.text[:150]}...")
        
        print("\n" + "-" * 80)
    
    # Summary statistics
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    total_evidence = sum(len(claim.evidence) for claim in claims)
    avg_evidence = total_evidence / len(claims) if claims else 0
    print(f"Total claims extracted: {len(claims)}")
    print(f"Total evidence retrieved: {total_evidence}")
    print(f"Average evidence per claim: {avg_evidence:.1f}")
    
    # Show which claims have overlapping sentences (sliding window effect)
    print(f"\n🔄 Sliding Window Effect:")
    for i, claim in enumerate(claims):
        overlaps = []
        for j, other_claim in enumerate(claims):
            if i != j:
                # Check if sentence indices overlap
                if set(claim.sentence_indices) & set(other_claim.sentence_indices):
                    overlaps.append(j)
        if overlaps:
            print(f"   Claim {i} overlaps with claims: {overlaps}")
    
    print("\n" + "=" * 80)
    print("✓ Demo completed!")
    print("=" * 80)


if __name__ == "__main__":
    main()
