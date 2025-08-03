import PyPDF2
import os
from pathlib import Path

def check_pdf_file(file_path: str) -> dict:
    """
    Check if PDF file is valid and not encrypted
    Returns: dict with status and message
    """
    try:
        # Check if file exists
        if not os.path.exists(file_path):
            return {
                "valid": False,
                "message": "❌ File does not exist"
            }
        
        # Check file extension
        if not file_path.lower().endswith('.pdf'):
            return {
                "valid": False,
                "message": "❌ File is not a PDF"
            }
        
        # Check file size
        file_size = os.path.getsize(file_path)
        if file_size == 0:
            return {
                "valid": False,
                "message": "❌ PDF file is empty"
            }
        
        # Try to open PDF with PyPDF2
        with open(file_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            
            # Check if PDF is encrypted
            if pdf_reader.is_encrypted:
                return {
                    "valid": False,
                    "message": "❌ PDF is password protected. Please remove the password and try again."
                }
            
            # Check number of pages
            num_pages = len(pdf_reader.pages)
            if num_pages == 0:
                return {
                    "valid": False,
                    "message": "❌ PDF has no readable pages"
                }
            
            # Try to extract text from first page
            try:
                first_page = pdf_reader.pages[0]
                text = first_page.extract_text()
                if not text.strip():
                    return {
                        "valid": False,
                        "message": "❌ PDF appears to be image-based or has no extractable text"
                    }
            except Exception as e:
                return {
                    "valid": False,
                    "message": f"❌ Cannot extract text from PDF: {str(e)}"
                }
            
            return {
                "valid": True,
                "message": f"✅ PDF is valid ({num_pages} pages, {file_size/1024/1024:.1f} MB)",
                "pages": num_pages,
                "size_mb": file_size/1024/1024
            }
            
    except Exception as e:
        return {
            "valid": False,
            "message": f"❌ Error checking PDF: {str(e)}"
        }

def fix_pdf_issues(file_path: str) -> dict:
    """
    Try to fix common PDF issues
    Returns: dict with status and suggestions
    """
    issues = []
    suggestions = []
    
    # Check file
    result = check_pdf_file(file_path)
    
    if not result["valid"]:
        issues.append(result["message"])
        
        # Provide suggestions based on error
        if "password" in result["message"].lower():
            suggestions.append("🔓 Remove password protection using Adobe Acrobat or online tools")
            suggestions.append("📝 Use 'Print to PDF' to create a new unprotected version")
        elif "image-based" in result["message"].lower():
            suggestions.append("🖼️ This PDF contains only images - use OCR tools to extract text")
            suggestions.append("📄 Convert to text-based PDF using Adobe Acrobat")
        elif "empty" in result["message"].lower():
            suggestions.append("📄 Try downloading the PDF again")
            suggestions.append("🔄 Check if the file was corrupted during upload")
    
    return {
        "issues": issues,
        "suggestions": suggestions,
        "can_fix": len(suggestions) > 0
    }

if __name__ == "__main__":
    # Example usage
    test_file = "test.pdf"
    if os.path.exists(test_file):
        result = check_pdf_file(test_file)
        print(f"File: {test_file}")
        print(f"Status: {result['message']}")
        
        if not result["valid"]:
            fix_result = fix_pdf_issues(test_file)
            print("\nSuggestions:")
            for suggestion in fix_result["suggestions"]:
                print(f"  {suggestion}")
    else:
        print("No test.pdf file found") 