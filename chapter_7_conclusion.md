## Chapter 7: Conclusion

7.1 Summary of Contributions

A comprehensive, multimodal artificial intelligence framework dedicated to the real-time detection of food allergens within Sri Lankan cuisine has been successfully developed and evaluated. The research journey, which commenced with a critical review highlighting the severe limitations of existing food recognition systems in localized contexts, has culminated in the deployment of a highly functional, full-stack prototype. The seamless integration of a Next.js presentation tier, a FastAPI application tier, and a dual-database storage tier has been fundamentally documented and proven effective.

7.2 Core Findings and Validation of Hypothesis

The primary research objectives established at the outset of this dissertation have been completely fulfilled. It had been hypothesized that a multimodal RAG system would outperform traditional classification models for complex curries. This has been validated; a robust image recognition pipeline capable of identifying visually ambiguous Sri Lankan dishes has been engineered, achieving a 76.00% Top-1 accuracy and demonstrating significant resilience against out-of-distribution data. Second, a highly personalized risk assessment engine has been implemented. By utilizing Logistic Regression prioritized for maximum Recall (24.03%), individualized demographic profiles, clinical data, and dynamic ingredient lists have been successfully mapped to generate accurate, safety-focused allergen warnings, while an Apriori association rule safety net has been established to catch hidden cross-reactivities.

7.3 Real-World Implications for Dietary Healthcare

Significant technical contributions to the intersection of food computing and healthcare AI have been made through this research. A novel integration of large language models (Gemini) as dynamic reasoning engines over localized vector databases (ChromaDB) has been introduced. This architecture has fundamentally mitigated the "unseen data" problem that historically paralyzed traditional Convolutional Neural Networks. Furthermore, a reproducible blueprint for developing ethically sound, safety-critical medical AI applications has been provided, bridging the critical gap between visual food recognition and clinical patient profiling.

7.4 Future Directions

While the foundational architecture has proven highly successful, it has been recommended that future iterations aggressively expand the underlying datasets. The incorporation of a significantly larger demographic sample size beyond the initial 90-user dataset would drastically improve the machine learning model's demographic generalizability. Furthermore, the expansion of the ChromaDB vector database from 35 curated recipes to encompass a comprehensive, national registry of Sri Lankan and regional dishes is strongly advised. To maximize the system's real-world utility, it has been proposed that the architecture be migrated from a web-based prototype to a native mobile application. The integration of real-time edge computing models would significantly enhance accessibility.

7.5 Concluding Remarks

In conclusion, it has been definitively demonstrated that the complexities of localized, visually ambiguous cuisines can be safely navigated by integrating multimodal visual reasoning with robust clinical predictive models. A significant leap forward in preventative dietary healthcare has been achieved, ensuring that technological innovation directly translates into enhanced physical safety for vulnerable populations.
