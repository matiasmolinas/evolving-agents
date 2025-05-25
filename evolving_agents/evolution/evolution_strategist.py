# evolving_agents/evolution/evolution_strategist.py

import logging
import asyncio
from typing import List, Dict, Any, Optional

from evolving_agents.monitoring.component_experience_tracker import ComponentExperienceTracker
from evolving_agents.auditing.library_auditor import LibraryAuditorAgent
# from evolving_agents.core.mongodb_client import MongoDBClient # For main example
# from evolving_agents.smart_library.smart_library import SmartLibrary # For main example

logger = logging.getLogger(__name__)

class EvolutionStrategistAgent:
    """
    Identifies components that are prime candidates for evolution
    based on audit results and experience data.
    """

    def __init__(self, experience_tracker: ComponentExperienceTracker, auditor: LibraryAuditorAgent):
        self.experience_tracker = experience_tracker
        self.auditor = auditor
        self.logger = logging.getLogger(__name__)
        if self.experience_tracker.experience_collection is None:
            self.logger.error("Experience collection in ComponentExperienceTracker is not initialized. Strategist may fail.")


    async def identify_evolution_candidates(self, top_n: int = 10) -> Dict[str, List[Dict[str, Any]]]:
        """
        Identifies and prioritizes components for evolution.
        """
        self.logger.info(f"Starting evolution candidate identification (top_n={top_n})...")

        # 1. Fetch Audit Data
        audit_results = await self.auditor.perform_audit()
        self.logger.info(f"Audit results received: { {k: len(v) for k, v in audit_results.items()} }")

        # 2. Fetch Additional Experience Data
        # Components with high failed_invocations
        high_failure_components = []
        if self.experience_tracker.experience_collection:
            try:
                # Using the placeholder method (assuming it's implemented or will be)
                # For now, let's use a direct query as per instructions.
                # high_failure_components = await self.experience_tracker.get_top_n_components_by_metric(
                #     metric="failed_invocations", n=top_n * 2, ascending=False # Fetch more to allow for overlap
                # )
                cursor = self.experience_tracker.experience_collection.find(
                    {"failed_invocations": {"$gt": 0}}
                ).sort("failed_invocations", -1).limit(top_n * 2) # pymongo.DESCENDING == -1
                async for doc in cursor:
                    high_failure_components.append(doc)
                self.logger.info(f"Fetched {len(high_failure_components)} components with high failed invocations.")
            except Exception as e:
                self.logger.error(f"Error fetching high failure components: {e}", exc_info=True)
        
        # Components with many distinct error types (more complex, placeholder for now)
        # This would typically require an aggregation query.
        # Example:
        # pipeline = [
        #     {"$match": {"error_types_summary": {"$exists": True, "$ne": {}}}},
        #     {"$addFields": {"num_error_types": {"$size": {"$objectToArray": "$error_types_summary"}}}},
        #     {"$sort": {"num_error_types": -1}},
        #     {"$limit": top_n * 2}
        # ]
        # diverse_error_components_cursor = self.experience_tracker.experience_collection.aggregate(pipeline)
        # async for doc in diverse_error_components_cursor: diverse_error_components.append(doc)


        # 3. Consolidate and Prioritize Candidates
        candidates: Dict[str, Dict[str, Any]] = {}

        def add_candidate(comp_id: str, name: Optional[str], record_type: Optional[str], reason: str, priority_bonus: int):
            if not comp_id: # Skip if comp_id is None or empty
                self.logger.warning(f"Skipping candidate with no ID. Reason: {reason}, Name: {name}")
                return
            if comp_id not in candidates:
                candidates[comp_id] = {
                    "id": comp_id,
                    "name": name or "N/A",
                    "record_type": record_type or "UNKNOWN",
                    "reasons": [],
                    "priority_score": 0
                }
            candidates[comp_id]["reasons"].append(reason)
            candidates[comp_id]["priority_score"] += priority_bonus
            if name and candidates[comp_id]["name"] == "N/A": # Update name if a better one comes along
                candidates[comp_id]["name"] = name
            if record_type and candidates[comp_id]["record_type"] == "UNKNOWN":
                candidates[comp_id]["record_type"] = record_type


        # Priority 1: Low success rate components (from auditor)
        for comp in audit_results.get("low_success_rate_components", []):
            reason_detail = comp.get('reason', f"Low success rate: {comp.get('success_rate', 'N/A')}")
            add_candidate(comp.get("id"), comp.get("name"), comp.get("record_type"), reason_detail, priority_bonus=20)

        # Priority 2: High failed invocations (direct query)
        for comp_exp in high_failure_components:
            reason = f"High failed invocations: {comp_exp.get('failed_invocations', 0)}"
            # Name might be in experience, or we might need to fetch it if not already covered by audit
            add_candidate(comp_exp.get("component_id"), comp_exp.get("name"), comp_exp.get("record_type"), reason, priority_bonus=15)

        # Priority 3: Potentially orphaned components (from auditor)
        for comp in audit_results.get("potentially_orphaned_components", []):
            add_candidate(comp.get("id"), comp.get("name"), comp.get("record_type"), comp.get('reason', "Potentially orphaned"), priority_bonus=10)
        
        # Priority 4: Unused components (from auditor) - lower priority for evolution, maybe for removal
        for comp in audit_results.get("unused_components", []):
            add_candidate(comp.get("id"), comp.get("name"), comp.get("record_type"), comp.get('reason', "Unused component"), priority_bonus=5)

        # Sort candidates by priority_score (descending), then by number of reasons (descending)
        sorted_candidates = sorted(
            candidates.values(),
            key=lambda x: (x["priority_score"], len(x["reasons"])),
            reverse=True
        )

        self.logger.info(f"Identified {len(sorted_candidates)} unique potential evolution candidates before limiting to top_n.")

        final_candidates = sorted_candidates[:top_n]

        self.logger.info(f"Returning {len(final_candidates)} prioritized evolution candidates.")
        return {"evolution_candidates": final_candidates}


async def main():
    # --- Mocking/Setup (replace with actual initialization) ---
    # This setup is illustrative and requires a running MongoDB instance
    # or more comprehensive mocking for ComponentExperienceTracker and SmartLibrary.

    from evolving_agents.core.mongodb_client import MongoDBClient
    from evolving_agents.smart_library.smart_library import SmartLibrary # For auditor

    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger.info("EvolutionStrategistAgent main example starting...")

    # ---- MongoDB Connection (replace with your actual URI and DB name) ----
    MONGODB_URI = "mongodb://localhost:27017/"
    DB_NAME = "evolving_agents_dev_strategist"
    
    try:
        mongodb_client = MongoDBClient(uri=MONGODB_URI, db_name=DB_NAME)
        # You might need an explicit connect method if your MongoDBClient has one
        # await mongodb_client.connect() 
        logger.info(f"Connected to MongoDB: {DB_NAME} (mocked or actual based on client)")

        # --- Instantiate dependencies ---
        # 1. SmartLibrary (needed by LibraryAuditorAgent)
        #    For this example, we'll use a mock version of SmartLibrary's data interactions
        #    by pre-populating its expected collection if it's empty or using mocks.
        smart_library_collection_name = "eat_library_components_strategist"
        smart_lib_mock_data = [
            {"id": "agent_001", "name": "Summarizer Agent", "type": "AGENT", "status": "active"},
            {"id": "tool_001", "name": "Calculator Tool", "type": "TOOL", "status": "active"},
            {"id": "agent_002", "name": "Data Analyst Agent", "type": "AGENT", "status": "active", "metadata": {"parent_id": "non_existent_parent"}}, # Orphaned
            {"id": "tool_002", "name": "File Writer Tool", "type": "TOOL", "status": "active"},
            {"id": "agent_003", "name": "Unused Agent", "type": "AGENT", "status": "active"},
        ]
        # Optional: Clear and insert mock data into SmartLibrary's collection for consistent testing
        # await mongodb_client.db[smart_library_collection_name].delete_many({})
        # if smart_library_collection_name not in await mongodb_client.db.list_collection_names() or await mongodb_client.db[smart_library_collection_name].count_documents({}) == 0 :
        #     await mongodb_client.db[smart_library_collection_name].insert_many(smart_lib_mock_data)
        #     logger.info(f"Inserted mock data into {smart_library_collection_name}")

        smart_library = SmartLibrary(mongodb_client=mongodb_client, components_collection_name=smart_library_collection_name)
        # Ensure SmartLibrary has some data if collection was empty
        if not await smart_library.export_records():
             logger.warning(f"SmartLibrary collection '{smart_library_collection_name}' is empty. Populating with mock data for example.")
             # Manually insert using the client if SmartLibrary doesn't have an import/insert method
             if smart_library_collection_name not in await mongodb_client.db.list_collection_names() or await mongodb_client.db[smart_library_collection_name].count_documents({}) == 0:
                 await mongodb_client.db[smart_library_collection_name].insert_many(smart_lib_mock_data)
                 logger.info(f"Inserted mock data into {smart_library_collection_name}")


        # 2. ComponentExperienceTracker
        experience_collection_name = "eat_component_experiences_strategist"
        exp_mock_data = [
            {"component_id": "agent_001", "name": "Summarizer Agent", "record_type": "AGENT", "total_invocations": 50, "successful_invocations": 20, "failed_invocations": 30, "error_types_summary": {"ValueError": 30}}, # Low success
            {"component_id": "tool_001", "name": "Calculator Tool", "record_type": "TOOL", "total_invocations": 5, "successful_invocations": 5, "failed_invocations": 0}, # Low invocations for success check
            {"component_id": "agent_002", "name": "Data Analyst Agent", "record_type": "AGENT", "total_invocations": 25, "successful_invocations": 23, "failed_invocations": 2}, # OK, but orphaned
            # tool_002 has no experience record -> unused by this metric
            # agent_003 has no experience record -> unused
        ]
        # Optional: Clear and insert mock data
        # await mongodb_client.db[experience_collection_name].delete_many({})
        # if experience_collection_name not in await mongodb_client.db.list_collection_names() or await mongodb_client.db[experience_collection_name].count_documents({}) == 0:
        #    await mongodb_client.db[experience_collection_name].insert_many(exp_mock_data)
        #    logger.info(f"Inserted mock data into {experience_collection_name}")
        
        experience_tracker = ComponentExperienceTracker(mongodb_client=mongodb_client, collection_name=experience_collection_name)
        # Ensure experience_tracker has some data if collection was empty
        # A bit harder to check without a direct "get_all" method, but we can try inserting if count is 0.
        if await experience_tracker.experience_collection.count_documents({}) == 0:
            logger.warning(f"ExperienceTracker collection '{experience_collection_name}' is empty. Populating with mock data for example.")
            await experience_tracker.experience_collection.insert_many(exp_mock_data)
            logger.info(f"Inserted mock data into {experience_collection_name}")


        # 3. LibraryAuditorAgent
        auditor = LibraryAuditorAgent(smart_library=smart_library, experience_tracker=experience_tracker)

        # 4. EvolutionStrategistAgent
        strategist = EvolutionStrategistAgent(experience_tracker=experience_tracker, auditor=auditor)

        # --- Run identification ---
        evolution_candidates_result = await strategist.identify_evolution_candidates(top_n=5)

        import json
        print("\n--- Evolution Candidates Result ---")
        print(json.dumps(evolution_candidates_result, indent=2))
        print("--- End of Evolution Candidates Result ---")

    except Exception as e:
        logger.error(f"Error in EvolutionStrategistAgent main example: {e}", exc_info=True)
    finally:
        if 'mongodb_client' in locals() and hasattr(mongodb_client, 'close'):
            # await mongodb_client.close() # If your client has an async close
            pass
        logger.info("EvolutionStrategistAgent main example finished.")


if __name__ == "__main__":
    # To run this example:
    # 1. Ensure MongoDB is running and accessible via MONGODB_URI.
    # 2. The example attempts to create/use specific collections for testing.
    #    You might want to clear these collections before runs for consistency.
    # 3. The mocking for SmartLibrary and ComponentExperienceTracker data is basic.
    #    It relies on direct DB inserts for the example.
    asyncio.run(main())
    pass
