import pandas as pd
import os
import re
from pathlib import Path
from collections import defaultdict

def parse_classes(classes_str):
    """
    Parse the Classes column to extract activity types and their time ranges.
    Example: "Sitting[0 to 1]; Sleeping[2 to 8]" -> {'Sitting': '0:1', 'Sleeping': '2:8'}
    """
    if pd.isna(classes_str) or not classes_str.strip():
        return {}
    
    activities = {}
    # Split by semicolon to get individual activities
    parts = classes_str.split(';')
    
    for part in parts:
        part = part.strip()
        # Match pattern: ActivityName[time range]
        match = re.match(r'([A-Za-z\s]+)\[(.*?)\]', part)
        if match:
            activity_name = match.group(1).strip()
            time_range = match.group(2).strip()
            # Convert "0 to 1" to "0:1"
            time_range = time_range.replace(' to ', ':')
            activities[activity_name] = time_range
    
    return activities

def create_activity_video_matrix(base_dir):
    """
    Create a matrix with activities as rows and video numbers as columns.
    """
    base_path = Path(base_dir)
    subject_dirs = sorted([d for d in base_path.iterdir() if d.is_dir() and d.name.startswith('Subject')])
    
    all_data = []
    
    for subject_dir in subject_dirs:
        subject_name = subject_dir.name
        
        for category in ['ADL', 'Fall']:
            csv_file = subject_dir / f'{category}.csv'
            if not csv_file.exists():
                continue
            
            try:
                df = pd.read_csv(csv_file)
                
                # Collect all activities and their time ranges per video
                video_activity_map = defaultdict(dict)  # {video_num: {activity: time_range}}
                all_activities = set()
                
                for _, row in df.iterrows():
                    video_name = row['File Name']
                    # Extract video number (e.g., "01" from "01.mp4")
                    video_num = video_name.replace('.mp4', '')
                    
                    classes = row.get('Classes', row.get(' Classes', ''))
                    activities = parse_classes(str(classes))
                    
                    for activity, time_range in activities.items():
                        video_activity_map[video_num][activity] = time_range
                        all_activities.add(activity)
                
                # Find all unique video numbers for this subject/category
                all_videos = sorted(video_activity_map.keys())
                
                # Create rows for each activity
                for activity in sorted(all_activities):
                    row_data = {
                        'Subject': subject_name,
                        'Category': category,
                        'Activity': activity
                    }
                    
                    # Add time ranges for each video
                    for video_num in all_videos:
                        col_name = f'{video_num}.mp4'
                        row_data[col_name] = video_activity_map[video_num].get(activity, '')
                    
                    all_data.append(row_data)
                    
            except Exception as e:
                print(f"Error reading {csv_file}: {e}")
    
    if all_data:
        df = pd.DataFrame(all_data)
        # Sort columns: Subject, Category, Activity, then video numbers
        fixed_cols = ['Subject', 'Category', 'Activity']
        video_cols = sorted([col for col in df.columns if col not in fixed_cols])
        df = df[fixed_cols + video_cols]
        return df
    
    return None

def create_separate_subject_sheets(base_dir):
    """
    Create separate sheets for each subject with activities as rows and videos as columns.
    """
    base_path = Path(base_dir)
    subject_dirs = sorted([d for d in base_path.iterdir() if d.is_dir() and d.name.startswith('Subject')])
    
    subject_dataframes = {}
    
    for subject_dir in subject_dirs:
        subject_name = subject_dir.name
        subject_data = []
        
        for category in ['ADL', 'Fall']:
            csv_file = subject_dir / f'{category}.csv'
            if not csv_file.exists():
                continue
            
            try:
                df = pd.read_csv(csv_file)
                
                # Collect all activities and their time ranges per video
                video_activity_map = defaultdict(dict)
                all_activities = set()
                
                for _, row in df.iterrows():
                    video_name = row['File Name']
                    video_num = video_name.replace('.mp4', '')
                    
                    classes = row.get('Classes', row.get(' Classes', ''))
                    activities = parse_classes(str(classes))
                    
                    for activity, time_range in activities.items():
                        video_activity_map[video_num][activity] = time_range
                        all_activities.add(activity)
                
                all_videos = sorted(video_activity_map.keys())
                
                # Add category header row
                header_row = {'Category': category, 'Activity': ''}
                for video_num in all_videos:
                    header_row[f'{video_num}.mp4'] = ''
                subject_data.append(header_row)
                
                # Create rows for each activity
                for activity in sorted(all_activities):
                    row_data = {
                        'Category': category,
                        'Activity': f'  {activity}'
                    }
                    
                    for video_num in all_videos:
                        col_name = f'{video_num}.mp4'
                        row_data[col_name] = video_activity_map[video_num].get(activity, '')
                    
                    subject_data.append(row_data)
                    
            except Exception as e:
                print(f"Error reading {csv_file}: {e}")
        
        if subject_data:
            subject_df = pd.DataFrame(subject_data)
            # Sort columns
            fixed_cols = ['Category', 'Activity']
            video_cols = sorted([col for col in subject_df.columns if col not in fixed_cols])
            subject_df = subject_df[fixed_cols + video_cols]
            subject_dataframes[subject_name] = subject_df
    
    return subject_dataframes

def main():
    base_dir = "./GMDCSA24-A-Dataset-for-Human-Fall-Detection-in-Videos"
    
    if not os.path.exists(base_dir):
        base_dir = "."
    
    print("="*80)
    print("CREATING ACTIVITY-VIDEO MATRIX")
    print("Activities as rows, Videos as columns")
    print("="*80)
    
    # Create combined matrix
    print("\nCreating combined matrix for all subjects...")
    combined_df = create_activity_video_matrix(base_dir)
    
    if combined_df is not None:
        # Save combined CSV
        combined_df.to_csv('Activity_Video_Matrix_All_Subjects.csv', index=False)
        print(f"✓ Created Activity_Video_Matrix_All_Subjects.csv")
        
        # Save combined Excel
        combined_df.to_excel('Activity_Video_Matrix_All_Subjects.xlsx', index=False, engine='openpyxl')
        print(f"✓ Created Activity_Video_Matrix_All_Subjects.xlsx")
    
    # Create separate sheets per subject
    print("\nCreating separate sheets for each subject...")
    subject_dfs = create_separate_subject_sheets(base_dir)
    
    if subject_dfs:
        # Save to Excel with multiple sheets
        with pd.ExcelWriter('Activity_Video_Matrix_By_Subject.xlsx', engine='openpyxl') as writer:
            for subject_name, df in subject_dfs.items():
                sheet_name = subject_name.replace(' ', '_')
                df.to_excel(writer, sheet_name=sheet_name, index=False)
                print(f"  ✓ Added sheet: {sheet_name}")
        
        print(f"✓ Created Activity_Video_Matrix_By_Subject.xlsx")
        
        # Save individual CSV files per subject
        for subject_name, df in subject_dfs.items():
            filename = f"{subject_name.replace(' ', '_')}_Activity_Matrix.csv"
            df.to_csv(filename, index=False)
            print(f"✓ Created {filename}")
    
    print("\n" + "="*80)
    print("FILES CREATED:")
    print("="*80)
    print("1. Activity_Video_Matrix_All_Subjects.csv")
    print("   - All subjects combined")
    print("   - Activities as rows, videos as columns")
    print("\n2. Activity_Video_Matrix_All_Subjects.xlsx")
    print("   - Excel version of the combined matrix")
    print("\n3. Activity_Video_Matrix_By_Subject.xlsx")
    print("   - Separate sheet for each subject")
    print("\n4. Individual CSV files:")
    print("   - Subject_1_Activity_Matrix.csv")
    print("   - Subject_2_Activity_Matrix.csv")
    print("   - Subject_3_Activity_Matrix.csv")
    print("   - Subject_4_Activity_Matrix.csv")
    
    # Show preview
    if combined_df is not None:
        print("\n" + "="*80)
        print("PREVIEW: Activity_Video_Matrix_All_Subjects.csv")
        print("="*80)
        print(combined_df.head(20).to_string(index=False, max_colwidth=15))
        
        print(f"\nTotal rows: {len(combined_df)}")
        print(f"Total columns: {len(combined_df.columns)}")
        print(f"Video columns: {len(combined_df.columns) - 3}")

if __name__ == "__main__":
    main()

