import streamlit as st
import json
import pandas as pd
from pathlib import Path
from typing import Dict, List, Any
import os

# Page configuration
st.set_page_config(
    page_title="Golden Answer Verification",
    page_icon="✅",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .correct-box {
        background-color: #d4edda;
        border: 2px solid #28a745;
        border-radius: 5px;
        padding: 15px;
        margin: 10px 0;
    }
    .incorrect-box {
        background-color: #f8d7da;
        border: 2px solid #dc3545;
        border-radius: 5px;
        padding: 15px;
        margin: 10px 0;
    }
    .revision-box {
        background-color: #fff3cd;
        border: 2px solid #ffc107;
        border-radius: 5px;
        padding: 15px;
        margin: 10px 0;
    }
    .stDataFrame {
        border: 2px solid #4CAF50;
        border-radius: 5px;
    }
</style>
""", unsafe_allow_html=True)


class GoldenAnswerVerifier:
    """Main application class for golden answer verification."""
    
    def __init__(self, base_dir: str):
        self.base_dir = Path(base_dir)
        self.qa_dir = self.base_dir / "qa_pairs" / 'en'
        self.qa_corrected_dir = self.base_dir / "qa_pairs_corrected"
        self.tables_dir = self.base_dir / "tables"
        self.verification_file = self.base_dir / "golden_answer_verification_results_v2.jsonl"
        
        # Create corrected directory if it doesn't exist
        self.qa_corrected_dir.mkdir(exist_ok=True)
        
        # Initialize session state
        if 'current_index' not in st.session_state:
            st.session_state.current_index = 0
        if 'filter_status' not in st.session_state:
            st.session_state.filter_status = "All"
        if 'filter_dataset' not in st.session_state:
            st.session_state.filter_dataset = "All"
        if 'verification_data' not in st.session_state:
            st.session_state.verification_data = None
        if 'changes_made' not in st.session_state:
            st.session_state.changes_made = {}
    
    def load_verification_results(self) -> List[Dict]:
        """Load verification results from JSONL file."""
        results = []
        if self.verification_file.exists():
            with open(self.verification_file, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        results.append(json.loads(line.strip()))
                    except Exception as e:
                        st.error(f"Error loading line: {e}")
        return results
    
    def load_table_data(self, question_id: str) -> Dict:
        """Load table data based on question ID."""
        # Extract table_id from question_id (format: table_id_q1, table_id_q2, etc.)
        table_id = "_".join(question_id.split("_")[:-1])
        table_file = self.tables_dir / f"{table_id}.json"
        
        if table_file.exists():
            with open(table_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        return {}
    
    def table_to_dataframe(self, table_data: Dict) -> pd.DataFrame:
        """Convert JSON table to pandas DataFrame.
        Assumes table_data has 'columns' and 'data' keys.
        """
        if not table_data or "columns" not in table_data or "data" not in table_data:
            return pd.DataFrame()
        
        headers = table_data["columns"]
        rows = table_data["data"]
        
        if not headers or not isinstance(headers, list):
            return pd.DataFrame()
        
        # Ensure rows are also a list of lists, even if empty
        if not rows or not isinstance(rows, list):
            rows = []
        
        return pd.DataFrame(rows, columns=headers)
    
    def update_qa_file(self, question_id: str, new_answer: List[str]):
        """Update the answer in the corrected QA file (creates copy if doesn't exist)."""
        table_id = "_".join(question_id.split("_")[:-1])
        original_qa_file = self.qa_dir / f"{table_id}_qa.json"
        corrected_qa_file = self.qa_corrected_dir / f"{table_id}_qa.json"
        
        if not original_qa_file.exists():
            st.error(f"Original QA file not found: {original_qa_file}")
            return False
        
        try:
            # If corrected file doesn't exist, copy from original
            if not corrected_qa_file.exists():
                with open(original_qa_file, 'r', encoding='utf-8') as f:
                    qa_data = json.load(f)
                st.info(f"📋 Creating new corrected file: {corrected_qa_file.name}")
            else:
                # Load existing corrected file
                with open(corrected_qa_file, 'r', encoding='utf-8') as f:
                    qa_data = json.load(f)
            
            # Find and update the specific question
            updated = False
            for item in qa_data:
                if item['question_id'] == question_id:
                    item['answer'] = new_answer
                    updated = True
                    break
            
            if updated:
                # Save to corrected file
                with open(corrected_qa_file, 'w', encoding='utf-8') as f:
                    json.dump(qa_data, f, indent=2, ensure_ascii=False)
                return True
            else:
                st.error(f"Question ID not found in QA file: {question_id}")
                return False
                
        except Exception as e:
            st.error(f"Error updating QA file: {e}")
            return False
    
    def get_dataset_type(self, question_id: str) -> str:
        """Extract dataset type from question_id (e.g., arxiv, finqa, wikitq)."""
        # Question IDs typically start with dataset name
        parts = question_id.lower().split('_')
        if len(parts) > 0:
            dataset = parts[0]
            # Common dataset prefixes
            if 'arxiv' in dataset:
                return 'arxiv'
            elif 'finqa' in dataset:
                return 'finqa'
            elif 'wiki' in dataset or 'wikitq' in dataset:
                return 'wikitq'
            elif 'tat' in dataset:
                return 'tat-qa'
            elif 'multihiertt' in dataset:
                return 'multihiertt'
        return 'unknown'
    
    def get_unique_datasets(self, data: List[Dict]) -> List[str]:
        """Get list of unique dataset types from verification data."""
        datasets = set()
        for item in data:
            dataset = self.get_dataset_type(item['question_id'])
            datasets.add(dataset)
        return sorted(list(datasets))
        """Filter verification data by status."""
        if status == "All":
            return data
        elif status == "Incorrect":
            return [d for d in data if d['is_golden_correct'] == 'Incorrect']
        elif status == "Needs_Revision":
            return [d for d in data if d['is_golden_correct'] == 'Needs_Revision']
        elif status == "Correct":
            return [d for d in data if d['is_golden_correct'] == 'Correct']
        elif status == "Error":
            return [d for d in data if d['is_golden_correct'] == 'Error']
        return data
    
    def display_status_badge(self, status: str):
        """Display status badge with appropriate styling."""
        if status == "Correct":
            st.markdown('<div class="correct-box">✅ <b>CORRECT</b></div>', unsafe_allow_html=True)
        elif status == "Incorrect":
            st.markdown('<div class="incorrect-box">❌ <b>INCORRECT</b></div>', unsafe_allow_html=True)
        elif status == "Needs_Revision":
            st.markdown('<div class="revision-box">⚠️ <b>NEEDS REVISION</b></div>', unsafe_allow_html=True)
        else:
            st.error(f"⚡ ERROR: {status}")
            
    def get_unique_datasets(self, data: List[Dict]) -> List[str]:
        """Get list of unique dataset types from verification data."""
        datasets = set()
        for item in data:
            dataset = self.get_dataset_type(item['question_id'])
            datasets.add(dataset)
        return sorted(list(datasets))

    def filter_data(self, data: List[Dict], status: str, dataset: str) -> List[Dict]:
        """Filter verification data by status and dataset."""
        filtered = data
        if status != "All":
            filtered = [d for d in filtered if d.get('is_golden_correct') == status]
        if dataset != "All":
            filtered = [d for d in filtered if self.get_dataset_type(d['question_id']) == dataset]
        return filtered
    
    def run(self):
        """Run the Streamlit application."""
        st.title("🔍 Golden Answer Verification System")
        st.markdown("---")
        
        # Sidebar
        with st.sidebar:
            st.header("⚙️ Configuration")
            
            # Load data button
            if st.button("🔄 Load/Refresh Data", use_container_width=True):
                with st.spinner("Loading verification results..."):
                    st.session_state.verification_data = self.load_verification_results()
                    st.session_state.current_index = 0
                st.success(f"Loaded {len(st.session_state.verification_data)} results")
            
            st.markdown("---")
            
            # Filter options
            st.subheader("🔍 Filters")
            
            # Status filter
            filter_status = st.selectbox(
                "Status",
                ["All", "Incorrect", "Needs_Revision", "Correct", "Error"],
                key="filter_select"
            )
            
            # Dataset filter
            if st.session_state.verification_data:
                datasets = self.get_unique_datasets(st.session_state.verification_data)
                filter_dataset = st.selectbox(
                    "Dataset Type",
                    ["All"] + datasets,
                    key="dataset_filter"
                )
            else:
                filter_dataset = "All"
            
            # Update filters and reset index if changed
            if (filter_status != st.session_state.filter_status or 
                filter_dataset != st.session_state.filter_dataset):
                st.session_state.filter_status = filter_status
                st.session_state.filter_dataset = filter_dataset
                st.session_state.current_index = 0
            
            st.markdown("---")
            
            # Statistics
            if st.session_state.verification_data:
                st.subheader("📊 Statistics")
                data = st.session_state.verification_data
                
                # Overall stats
                total = len(data)
                correct = sum(1 for d in data if d['is_golden_correct'] == 'Correct')
                incorrect = sum(1 for d in data if d['is_golden_correct'] == 'Incorrect')
                needs_rev = sum(1 for d in data if d['is_golden_correct'] == 'Needs_Revision')
                errors = sum(1 for d in data if d['is_golden_correct'] == 'Error')
                
                st.markdown("**Overall:**")
                st.metric("Total", total)
                st.metric("✅ Correct", f"{correct} ({correct/total*100:.1f}%)")
                st.metric("❌ Incorrect", f"{incorrect} ({incorrect/total*100:.1f}%)")
                st.metric("⚠️ Needs Revision", f"{needs_rev} ({needs_rev/total*100:.1f}%)")
                st.metric("⚡ Errors", errors)
                
                # Dataset breakdown
                st.markdown("**By Dataset:**")
                datasets = self.get_unique_datasets(data)
                for dataset in datasets:
                    dataset_data = [d for d in data if self.get_dataset_type(d['question_id']) == dataset]
                    count = len(dataset_data)
                    st.text(f"{dataset}: {count} ({count/total*100:.1f}%)")
                
                st.markdown("---")
                st.metric("📝 Changes Made", len(st.session_state.changes_made))
                
                # Info about corrected files location
                st.markdown("---")
                st.info(f"💾 **Corrected files saved to:**\n`qa_pairs_corrected/`")
        
        # Main content
        if st.session_state.verification_data is None:
            st.info("👈 Click 'Load/Refresh Data' in the sidebar to begin")
            return
        
        # Filter data
        filtered_data = self.filter_data(
            st.session_state.verification_data,
            st.session_state.filter_status,
            st.session_state.filter_dataset
        )
        
        if not filtered_data:
            st.warning(f"No items found with Status: {st.session_state.filter_status}, Dataset: {st.session_state.filter_dataset}")
            return
        
        # Navigation
        col1, col2, col3 = st.columns([1, 3, 1])
        with col1:
            if st.button("⬅️ Previous", disabled=st.session_state.current_index == 0):
                st.session_state.current_index -= 1
                st.rerun()
        
        with col2:
            st.markdown(f"<h3 style='text-align: center;'>Item {st.session_state.current_index + 1} of {len(filtered_data)}</h3>", unsafe_allow_html=True)
        
        with col3:
            if st.button("Next ➡️", disabled=st.session_state.current_index >= len(filtered_data) - 1):
                st.session_state.current_index += 1
                st.rerun()
        
        st.markdown("---")
        
        # Get current item
        current_item = filtered_data[st.session_state.current_index]
        
        # Display verification status
        col1, col2 = st.columns([1, 3])
        with col1:
            self.display_status_badge(current_item['is_golden_correct'])
        with col2:
            st.metric("Confidence", current_item['confidence'])
        
        # Question details
        st.header("📋 Question Details")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.info(f"**Question ID:** {current_item['question_id']}")
        with col2:
            st.info(f"**Question Type:** {current_item['question_type']}")
        with col3:
            dataset_type = self.get_dataset_type(current_item['question_id'])
            st.info(f"**Dataset:** {dataset_type}")
        
        st.markdown(f"**Question:** {current_item['question']}")
        
        st.markdown("---")
        
        # Table display
        st.header("📊 Table Data")
        table_data = self.load_table_data(current_item['question_id'])
        df = self.table_to_dataframe(table_data)
        
        if not df.empty:
            st.dataframe(df, use_container_width=True, height=300)
        else:
            st.warning("Table data not available")
        
        st.markdown("---")
        
        # Model responses
        st.header("🤖 Model Responses")
        for i, response in enumerate(current_item.get('model_responses', [])):
            with st.expander(f"Model {i+1}: {response.get('model_name', 'Unknown')}"):
                st.markdown(response.get('model_response', 'N/A'))
        
        st.markdown("---")
        
        # Verification analysis
        st.header("🔍 Verification Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Golden Answer")
            # Handle nested list format [["ans"]]
            golden_answer = current_item['golden_answer']
            if isinstance(golden_answer, list) and len(golden_answer) > 0:
                if isinstance(golden_answer[0], list):
                    # Nested list format [["ans"]]
                    st.json(golden_answer)
                else:
                    # Simple list format ["ans"]
                    st.json(golden_answer)
            else:
                st.json(golden_answer)
        
        with col2:
            st.subheader("Suggested Corrected Answer")
            # Handle nested list format [["ans"]]
            corrected_answer = current_item['corrected_answer']
            if isinstance(corrected_answer, list) and len(corrected_answer) > 0:
                if isinstance(corrected_answer[0], list):
                    # Nested list format [["ans"]]
                    st.json(corrected_answer)
                else:
                    # Simple list format ["ans"]
                    st.json(corrected_answer)
            else:
                st.json(corrected_answer)
        
        # Reasoning
        st.subheader("💭 Reasoning")
        st.text_area(
            "Detailed Analysis",
            value=current_item['reasoning'],
            height=200,
            disabled=True,
            key=f"reasoning_{st.session_state.current_index}"
        )
        
        # Evidence summary
        st.subheader("📑 Evidence Summary")
        st.info(current_item['evidence_summary'])
        
        st.markdown("---")
        
        # Answer correction interface
        st.header("✏️ Correct Answer")
        
        # Parse answer for editing - handle nested lists
        current_answer = current_item['corrected_answer']
        if isinstance(current_answer, list):
            # Check if it's nested list format [["ans"]]
            if len(current_answer) > 0 and isinstance(current_answer[0], list):
                answer_str = json.dumps(current_answer, indent=2)
            else:
                # Convert simple list to nested format for consistency
                answer_str = json.dumps([current_answer], indent=2)
        else:
            answer_str = json.dumps([[str(current_answer)]], indent=2)
        
        st.info("💡 **Note:** Answers should be in nested list format: `[[\"answer\"]]` or `[[\"ans1\", \"ans2\"]]`")
        
        # Edit answer
        new_answer_str = st.text_area(
            "Edit the correct answer (nested JSON list format: [[\"answer\"]])",
            value=answer_str,
            height=150,
            key=f"answer_edit_{st.session_state.current_index}"
        )
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("💾 Save Correction", type="primary", use_container_width=True):
                try:
                    # Parse the new answer
                    new_answer = json.loads(new_answer_str)
                    
                    # Validate nested list format
                    if not isinstance(new_answer, list):
                        st.error("Answer must be a list")
                    elif len(new_answer) == 0:
                        st.error("Answer cannot be empty")
                    elif not isinstance(new_answer[0], list):
                        st.warning("Converting to nested list format: [[...]]")
                        new_answer = [new_answer]
                    
                    # Update the QA file (creates corrected copy)
                    if self.update_qa_file(current_item['question_id'], new_answer):
                        st.success("✅ Answer saved to corrected QA file!")
                        st.session_state.changes_made[current_item['question_id']] = {
                            'old': current_item['golden_answer'],
                            'new': new_answer,
                            'timestamp': pd.Timestamp.now().isoformat(),
                            'file': f"qa_pairs_corrected/{('_'.join(current_item['question_id'].split('_')[:-1]))}_qa.json",
                            'dataset': self.get_dataset_type(current_item['question_id'])
                        }
                        
                        # Auto-advance to next item
                        if st.session_state.current_index < len(filtered_data) - 1:
                            st.session_state.current_index += 1
                            st.rerun()
                    else:
                        st.error("Failed to save correction")
                        
                except json.JSONDecodeError as e:
                    st.error(f"Invalid JSON format: {e}")
                except Exception as e:
                    st.error(f"Error: {e}")
        
        with col2:
            if st.button("✅ Accept as Correct", use_container_width=True):
                st.success("Marked as correct (no changes needed)")
                if st.session_state.current_index < len(filtered_data) - 1:
                    st.session_state.current_index += 1
                    st.rerun()
        
        with col3:
            if st.button("⏭️ Skip", use_container_width=True):
                if st.session_state.current_index < len(filtered_data) - 1:
                    st.session_state.current_index += 1
                    st.rerun()
        
        # Export changes log
        if st.session_state.changes_made:
            st.markdown("---")
            if st.button("📥 Export Changes Log", use_container_width=True):
                changes_file = self.base_dir / "verification_changes_log.json"
                with open(changes_file, 'w', encoding='utf-8') as f:
                    json.dump(st.session_state.changes_made, f, indent=2, ensure_ascii=False)
                st.success(f"Changes log exported to: {changes_file}")


def main():
    """Main entry point."""
    # Configuration
    BASE_DIR = "data/processed/"
    
    # Check if directory exists
    if not Path(BASE_DIR).exists():
        st.error(f"Base directory not found: {BASE_DIR}")
        st.info("Please update BASE_DIR in the script to match your data location")
        return
    
    # Initialize and run app
    app = GoldenAnswerVerifier(BASE_DIR)
    app.run()


if __name__ == "__main__":
    main()