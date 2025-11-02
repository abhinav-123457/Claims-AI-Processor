from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import torch
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import io
import base64
import json
from datetime import datetime
import os
import tempfile
from reportlab.lib.pagesizes import letter, A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image as ReportLabImage
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib import colors
import logging
import uuid

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

class SmartAIDetector:
    def __init__(self):
        self.model_name = "umm-maybe/AI-image-detector"
        self.model = None
        self.processor = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.load_model()
    
    def load_model(self):
        """Load the Hugging Face AI image detection model"""
        try:
            logger.info(f"Loading AI detection model: {self.model_name}")
            from transformers import AutoImageProcessor, AutoModelForImageClassification
            self.processor = AutoImageProcessor.from_pretrained(self.model_name)
            self.model = AutoModelForImageClassification.from_pretrained(self.model_name)
            self.model.to(self.device)
            self.model.eval()
            logger.info("AI detection model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load AI detection model: {e}")
            self.model = None
            self.processor = None
    
    def detect_ai_image(self, image):
        """Smart AI image detection for claims processing"""
        try:
            # If model failed to load, return default result
            if self.model is None:
                return {
                    'is_ai_generated': False,
                    'confidence': 0.0,
                    'risk_level': 'low',
                    'reasons': ['AI detection model not available'],
                    'model_used': 'fallback'
                }
            
            # Ensure image is in RGB mode
            if isinstance(image, np.ndarray):
                pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            else:
                pil_image = image.convert('RGB')
            
            # Method 1: Hugging Face Model Detection
            hf_result = self.huggingface_detection(pil_image)
            
            # Method 2: Traditional Analysis (for verification)
            traditional_result = self.traditional_analysis(pil_image)
            
            # Method 3: Real Image Indicators (to reduce false positives)
            real_image_indicators = self.check_real_image_indicators(pil_image)
            
            # Adjust confidence based on real image indicators
            adjusted_hf_confidence = self.adjust_confidence(hf_result['confidence'], real_image_indicators)
            
            # Only use traditional analysis to INCREASE confidence, not decrease
            final_confidence = max(adjusted_hf_confidence, traditional_result['confidence'] * 0.3)
            
            # Conservative threshold for claims processing
            is_ai_generated = final_confidence > 0.35
            
            result = {
                'is_ai_generated': is_ai_generated,
                'confidence': final_confidence,
                'huggingface_confidence': hf_result['confidence'],
                'adjusted_confidence': adjusted_hf_confidence,
                'traditional_confidence': traditional_result['confidence'],
                'real_indicators_score': real_image_indicators['score'],
                'reasons': hf_result.get('reasons', []) + traditional_result.get('reasons', []) + real_image_indicators.get('reasons', []),
                'model_used': self.model_name,
                'risk_level': self.calculate_risk_level(final_confidence, is_ai_generated)
            }
            
            logger.info(f"Claims AI detection - Final: {final_confidence:.3f}, Risk: {result['risk_level']}, Decision: {'AI' if is_ai_generated else 'Real'}")
            return result
            
        except Exception as e:
            logger.error(f"Smart AI detection error: {e}")
            return {
                'is_ai_generated': False,
                'confidence': 0.0,
                'risk_level': 'low',
                'error': str(e)
            }
    
    def calculate_risk_level(self, confidence, is_ai_generated):
        """Calculate risk level for claims processing"""
        if is_ai_generated:
            if confidence > 0.8:
                return 'very_high'
            elif confidence > 0.6:
                return 'high'
            elif confidence > 0.4:
                return 'medium'
            else:
                return 'low'
        else:
            if confidence > 0.7:
                return 'verified_real'
            else:
                return 'likely_real'
    
    def huggingface_detection(self, image):
        """Hugging Face model detection"""
        try:
            # Preprocess image for the model
            inputs = self.processor(images=image, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Run inference
            with torch.no_grad():
                outputs = self.model(**inputs)
                predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
            
            # Get results - Model classes: 0 = AI-generated, 1 = Real
            predicted_class_idx = predictions.argmax().item()
            confidence = predictions[0][predicted_class_idx].item()
            
            is_ai_generated = predicted_class_idx == 0
            ai_confidence = confidence if is_ai_generated else 1 - confidence
            
            reasons = []
            if is_ai_generated:
                if ai_confidence > 0.9:
                    reasons.append("Very strong AI image patterns detected")
                elif ai_confidence > 0.8:
                    reasons.append("Strong AI image patterns detected")
                elif ai_confidence > 0.7:
                    reasons.append("AI image patterns detected")
                elif ai_confidence > 0.5:
                    reasons.append("Possible AI generation indicators")
            
            return {
                'is_ai_generated': is_ai_generated,
                'confidence': ai_confidence,
                'reasons': reasons
            }
            
        except Exception as e:
            logger.error(f"Hugging Face detection error: {e}")
            return {'is_ai_generated': False, 'confidence': 0.0, 'reasons': []}
    
    def check_real_image_indicators(self, image):
        """Check for indicators that suggest a real photograph"""
        try:
            score = 0.0
            reasons = []
            
            # Convert to numpy for analysis
            img_array = np.array(image)
            height, width = img_array.shape[:2]
            
            # 1. Check for natural noise and grain
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
            noise_variance = np.var(gray)
            if 100 < noise_variance < 1000:  # Natural noise range
                score += 0.3
                reasons.append("Natural image noise pattern")
            
            # 2. Check for realistic edge distribution
            edges = cv2.Canny(gray, 50, 150)
            edge_density = np.sum(edges > 0) / edges.size
            if 0.01 < edge_density < 0.2:  # Realistic edge density
                score += 0.2
                reasons.append("Realistic edge distribution")
            
            # 3. Check color variance (real photos have more color variation)
            color_variance = np.var(img_array, axis=(0, 1))
            avg_color_variance = np.mean(color_variance)
            if avg_color_variance > 500:  # Good color variation
                score += 0.2
                reasons.append("Natural color variation")
            
            # 4. Check for realistic texture
            sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
            sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
            texture_variance = np.var(sobel_x) + np.var(sobel_y)
            if texture_variance > 1000:  # Good texture detail
                score += 0.2
                reasons.append("Detailed texture patterns")
            
            # 5. Check image size and aspect ratio
            if width > 800 and height > 600:  # Reasonable size
                score += 0.1
                reasons.append("Appropriate image size for claims")
            
            return {
                'score': min(score, 1.0),
                'reasons': reasons
            }
            
        except Exception as e:
            logger.error(f"Real image indicators error: {e}")
            return {'score': 0.0, 'reasons': []}
    
    def adjust_confidence(self, hf_confidence, real_indicators):
        """Adjust HF confidence based on real image indicators"""
        # If real indicators are strong, reduce AI confidence
        adjustment = real_indicators['score'] * 0.3  # Reduce up to 30%
        adjusted_confidence = max(0.0, hf_confidence - adjustment)
        
        logger.info(f"Confidence adjustment: {hf_confidence:.3f} -> {adjusted_confidence:.3f} (real score: {real_indicators['score']:.2f})")
        return adjusted_confidence
    
    def traditional_analysis(self, image):
        """Traditional computer vision analysis for claims verification"""
        try:
            scores = []
            reasons = []
            
            # Only use traditional analysis for high-confidence AI indicators
            img_array = np.array(image)
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
            
            # 1. Check for extremely uniform noise (strong AI indicator)
            noise_variance = np.var(gray)
            if noise_variance < 20:  # Very uniform noise
                scores.append(0.8)
                reasons.append("Extremely uniform noise patterns - potential AI generation")
            elif noise_variance < 50:
                scores.append(0.5)
                reasons.append("Uniform noise patterns detected")
            else:
                scores.append(0.0)
            
            # 2. Check for metadata clues
            metadata_score = self.metadata_analysis(image)
            if metadata_score > 0.5:
                scores.append(metadata_score)
                reasons.append("AI generation metadata found")
            
            # 3. Check for compression artifacts (real images often have more)
            compression_score = self.check_compression_artifacts(img_array)
            if compression_score > 0:
                scores.append(compression_score)
                reasons.append("Compression pattern analysis")
            
            avg_score = sum(scores) / len(scores) if scores else 0.0
            
            return {
                'is_ai_generated': avg_score > 0.6,
                'confidence': avg_score,
                'reasons': reasons
            }
            
        except Exception as e:
            logger.error(f"Traditional analysis error: {e}")
            return {'is_ai_generated': False, 'confidence': 0.0, 'reasons': []}
    
    def check_compression_artifacts(self, img_array):
        """Check for JPEG compression artifacts common in real photos"""
        try:
            # Convert to YUV color space
            yuv = cv2.cvtColor(img_array, cv2.COLOR_RGB2YUV)
            # Calculate variance in Y channel (luminance)
            y_variance = np.var(yuv[:,:,0])
            
            if y_variance > 1000:  # High variance suggests real photo compression
                return 0.0
            else:
                return 0.3  # Low variance might indicate AI generation
        except:
            return 0.0
    
    def metadata_analysis(self, image):
        """Check metadata for AI generation clues"""
        try:
            score = 0.0
            
            # Check PNG text chunks
            if hasattr(image, 'text') and image.text:
                for key, value in image.text.items():
                    if isinstance(value, str):
                        value_lower = value.lower()
                        ai_indicators = [
                            'stable diffusion', 'midjourney', 'dall-e', 'dall·e', 
                            'generative', 'ai generated', 'ai-generated', 'prompt:',
                            'negative prompt:', 'steps:', 'sampler:', 'cfg scale:',
                            'novelai', 'leonardo', 'playground'
                        ]
                        
                        for indicator in ai_indicators:
                            if indicator in value_lower:
                                return 0.9  # Very high confidence
            
            return score
            
        except Exception as e:
            return 0.0

class PDFGenerator:
    """PDF generation for claims reports"""
    
    def __init__(self):
        self.styles = getSampleStyleSheet()
        self.setup_custom_styles()
    
    def setup_custom_styles(self):
        """Setup custom styles for claims reports"""
        # Title style
        self.title_style = ParagraphStyle(
            'CustomTitle',
            parent=self.styles['Heading1'],
            fontSize=18,
            spaceAfter=30,
            textColor=colors.HexColor('#2E7D32'),
            alignment=1  # Center
        )
        
        # Heading style
        self.heading_style = ParagraphStyle(
            'CustomHeading',
            parent=self.styles['Heading2'],
            fontSize=14,
            spaceAfter=12,
            spaceBefore=12,
            textColor=colors.HexColor('#1976D2')
        )
        
        # Normal style
        self.normal_style = ParagraphStyle(
            'CustomNormal',
            parent=self.styles['Normal'],
            fontSize=10,
            spaceAfter=6
        )
        
        # Risk style
        self.risk_style = ParagraphStyle(
            'RiskStyle',
            parent=self.styles['Normal'],
            fontSize=10,
            textColor=colors.red,
            backColor=colors.yellow,
            spaceAfter=6
        )
    
    def generate_claims_report(self, analysis_result, currency='USD'):
        """Generate comprehensive claims report PDF"""
        try:
            # Create a buffer for the PDF
            buffer = io.BytesIO()
            
            # Create PDF document
            doc = SimpleDocTemplate(
                buffer,
                pagesize=A4,
                rightMargin=72,
                leftMargin=72,
                topMargin=72,
                bottomMargin=18
            )
            
            # Build story (content)
            story = []
            
            # Add title
            story.append(Paragraph("INSURANCE CLAIM ASSESSMENT REPORT", self.title_style))
            story.append(Spacer(1, 20))
            
            # Claim Information
            story.append(Paragraph("CLAIM INFORMATION", self.heading_style))
            claim_data = [
                ['Claim ID:', analysis_result.get('claim_id', 'N/A')],
                ['Assessment Date:', datetime.now().strftime('%Y-%m-%d %H:%M:%S')],
                ['Total Estimated Cost:', f"{self.format_currency(analysis_result.get('total_cost', 0), currency)}"],
                ['Damage Count:', str(analysis_result.get('damage_summary', {}).get('total_count', 0))],
                ['AI Fraud Risk:', analysis_result.get('ai_validation', {}).get('risk_level', 'low').upper()]
            ]
            
            claim_table = Table(claim_data, colWidths=[2*inch, 3*inch])
            claim_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#E3F2FD')),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.HexColor('#1976D2')),
                ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, -1), 10),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                ('GRID', (0, 0), (-1, -1), 1, colors.black)
            ]))
            story.append(claim_table)
            story.append(Spacer(1, 20))
            
            # AI Validation Results
            ai_validation = analysis_result.get('ai_validation', {})
            story.append(Paragraph("AI FRAUD DETECTION ANALYSIS", self.heading_style))
            
            ai_data = [
                ['AI Image Detection:', 'YES' if ai_validation.get('is_ai_generated') else 'NO'],
                ['Confidence Level:', f"{ai_validation.get('confidence', 0) * 100:.1f}%"],
                ['Risk Level:', ai_validation.get('risk_level', 'low').upper()],
                ['Status:', 'HIGH RISK - VERIFICATION REQUIRED' if ai_validation.get('is_ai_generated') else 'LOW RISK - CLEAR']
            ]
            
            ai_table = Table(ai_data, colWidths=[2*inch, 3*inch])
            ai_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#FFEBEE') if ai_validation.get('is_ai_generated') else colors.HexColor('#E8F5E8')),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.red if ai_validation.get('is_ai_generated') else colors.green),
                ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, -1), 10),
                ('GRID', (0, 0), (-1, -1), 1, colors.black)
            ]))
            story.append(ai_table)
            story.append(Spacer(1, 12))
            
            # Damage Breakdown
            story.append(Paragraph("DAMAGE ASSESSMENT BREAKDOWN", self.heading_style))
            
            damage_data = [['Type', 'Severity', 'Priority', 'Confidence', 'Estimated Cost']]
            detections = analysis_result.get('detections', [])
            
            for detection in detections:
                cost = detection.get('cost_estimation', {}).get('base', 0)
                damage_data.append([
                    detection.get('type', '').replace('_', ' ').title(),
                    detection.get('severity', '').upper(),
                    detection.get('repair_priority', '').upper(),
                    f"{detection.get('confidence', 0) * 100:.1f}%",
                    self.format_currency(cost, currency)
                ])
            
            if len(damage_data) > 1:
                damage_table = Table(damage_data, colWidths=[1.2*inch, 0.8*inch, 0.8*inch, 0.8*inch, 1.2*inch])
                damage_table.setStyle(TableStyle([
                    ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1976D2')),
                    ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                    ('ALIGN', (0, 0), (-1, 0), 'CENTER'),
                    ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                    ('FONTSIZE', (0, 0), (-1, -1), 8),
                    ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                    ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                    ('GRID', (0, 0), (-1, -1), 1, colors.black),
                    ('ALIGN', (0, 1), (-1, -1), 'CENTER')
                ]))
                story.append(damage_table)
            else:
                story.append(Paragraph("No damages detected", self.normal_style))
            
            story.append(Spacer(1, 20))
            
            # Cost Summary
            story.append(Paragraph("COST ESTIMATION SUMMARY", self.heading_style))
            
            cost_summary = analysis_result.get('cost_estimation', {})
            total_cost = cost_summary.get('total_usd', 0)
            
            cost_data = [
                ['Total Estimated Cost:', self.format_currency(total_cost, currency)],
                ['Cost Range:', f"{self.format_currency(cost_summary.get('range_usd', {}).get('min', 0), currency)} - {self.format_currency(cost_summary.get('range_usd', {}).get('max', 0), currency)}"],
                ['Auto-Approval Eligible:', 'YES' if analysis_result.get('claims_processing', {}).get('auto_approval_eligible') else 'NO'],
                ['Estimated Processing Time:', f"{analysis_result.get('claims_processing', {}).get('estimated_processing_time', 3)} days"]
            ]
            
            cost_table = Table(cost_data, colWidths=[2.5*inch, 2.5*inch])
            cost_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#4CAF50')),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, -1), 10),
                ('BACKGROUND', (0, 1), (-1, -1), colors.HexColor('#E8F5E8')),
                ('GRID', (0, 0), (-1, -1), 1, colors.black)
            ]))
            story.append(cost_table)
            
            # Processing Recommendations
            recommendations = analysis_result.get('claims_processing', {}).get('recommendations', [])
            if recommendations:
                story.append(Spacer(1, 20))
                story.append(Paragraph("PROCESSING RECOMMENDATIONS", self.heading_style))
                
                for i, recommendation in enumerate(recommendations, 1):
                    story.append(Paragraph(f"{i}. {recommendation}", self.normal_style))
            
            # Footer
            story.append(Spacer(1, 30))
            story.append(Paragraph("Generated by Claims AI Processor - Automated Claims Assessment System", 
                                 ParagraphStyle('Footer', parent=self.styles['Normal'], fontSize=8, alignment=1)))
            
            # Build PDF
            doc.build(story)
            
            # Get PDF data
            pdf_data = buffer.getvalue()
            buffer.close()
            
            return pdf_data
            
        except Exception as e:
            logger.error(f"PDF generation error: {e}")
            raise e
    
    def format_currency(self, amount, currency):
        """Format currency amount based on currency code"""
        try:
            # Currency symbols
            symbols = {
                'USD': '$',
                'EUR': '€',
                'GBP': '£',
                'INR': '₹',
                'JPY': '¥',
                'CAD': 'C$',
                'AUD': 'A$'
            }
            
            symbol = symbols.get(currency, '$')
            
            # Format number with commas
            formatted_amount = f"{amount:,.0f}"
            
            return f"{symbol}{formatted_amount}"
            
        except:
            return f"${amount:,.0f}"

class ClaimsDamageAnalyzer:
    def __init__(self):
        # Try to load YOLO model, but don't fail if not available
        self.model = None
        self.model_loaded = False
        
        try:
            from ultralytics import YOLO
            model_paths = ['best.pt', './best.pt', 'model/best.pt', './model/best.pt']
            model_path = None
            for path in model_paths:
                if os.path.exists(path):
                    model_path = path
                    logger.info(f"Found model at: {path}")
                    break
            
            if model_path:
                self.model = YOLO(model_path)
                self.model_loaded = True
                logger.info("YOLO model loaded successfully")
            else:
                logger.warning("No YOLO model found. Using demo mode.")
        except ImportError:
            logger.warning("Ultralytics YOLO not available. Using demo mode.")
        except Exception as e:
            logger.warning(f"Failed to load YOLO model: {e}. Using demo mode.")

        # Initialize Smart AI Image Detector
        try:
            self.ai_detector = SmartAIDetector()
            self.ai_detection_available = True
            logger.info("Smart AI detector initialized successfully")
        except Exception as e:
            logger.warning(f"Smart AI detector failed: {e}")
            self.ai_detection_available = False
        
        # Initialize PDF Generator
        self.pdf_generator = PDFGenerator()
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Damage classes for claims processing
        self.damage_classes = ['dent', 'scratch', 'crack', 'broken_glass', 'broken_light', 'flat_tire']
        
        # Base repair costs in USD with severity levels
        self.repair_costs = {
            'dent': {'min': 150, 'max': 800, 'base': 300, 'severity': 'medium'},
            'scratch': {'min': 100, 'max': 400, 'base': 200, 'severity': 'low'},
            'crack': {'min': 300, 'max': 1200, 'base': 500, 'severity': 'high'},
            'broken_glass': {'min': 400, 'max': 1500, 'base': 800, 'severity': 'high'},
            'broken_light': {'min': 200, 'max': 800, 'base': 400, 'severity': 'medium'},
            'flat_tire': {'min': 150, 'max': 500, 'base': 250, 'severity': 'medium'}
        }
        
        # Currency conversion rates
        self.currency_rates = {
            'USD': 1.0,
            'EUR': 0.85,
            'GBP': 0.73,
            'INR': 83.0,
            'JPY': 150.0,
            'CAD': 1.35,
            'AUD': 1.50
        }
        
        # Claims processing thresholds
        self.claims_thresholds = {
            'auto_approve_max': 1000,  # USD
            'ai_detection_high_risk_threshold': 0.7,
            'ai_detection_medium_risk_threshold': 0.4
        }

    def preprocess_image(self, image_data):
        """Convert base64 image to OpenCV format for claims processing"""
        try:
            if isinstance(image_data, str) and image_data.startswith('data:image'):
                image_data = image_data.split(',')[1]
            
            image_bytes = base64.b64decode(image_data)
            image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
            cv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            return cv_image, image
            
        except Exception as e:
            logger.error(f"Error preprocessing image: {e}")
            raise e

    def validate_image_for_claims(self, image):
        """Smart image validation for insurance claims"""
        validation_result = {
            'is_valid': True,
            'is_ai_generated': False,
            'ai_confidence': 0.0,
            'risk_level': 'low',
            'errors': [],
            'warnings': [],
            'recommendations': [],
            'ai_detection': None,
            'detection_method': 'smart_claims'
        }
        
        try:
            # Smart AI detection for claims
            if self.ai_detection_available:
                ai_detection = self.ai_detector.detect_ai_image(image)
                validation_result['ai_detection'] = ai_detection
                validation_result['is_ai_generated'] = ai_detection.get('is_ai_generated', False)
                validation_result['ai_confidence'] = ai_detection.get('confidence', 0.0)
                validation_result['risk_level'] = ai_detection.get('risk_level', 'low')
                
                # Add AI detection info to warnings
                if ai_detection.get('is_ai_generated', False):
                    confidence_pct = ai_detection.get('confidence', 0) * 100
                    risk_level = ai_detection.get('risk_level', 'medium')
                    
                    if risk_level in ['high', 'very_high']:
                        validation_result['warnings'].append(
                            f"HIGH RISK: Potential AI-Generated Image ({confidence_pct:.1f}% confidence)"
                        )
                        validation_result['recommendations'].append("Manual review recommended for high-risk claims")
                    else:
                        validation_result['warnings'].append(
                            f"AI-Generated Image Detected ({confidence_pct:.1f}% confidence)"
                        )
                    
                    # Add specific reasons
                    if ai_detection.get('reasons'):
                        for reason in ai_detection['reasons']:
                            validation_result['warnings'].append(f"• {reason}")
                else:
                    # Real image verification
                    if ai_detection.get('confidence', 0) > 0.7:
                        validation_result['recommendations'].append("Image verified as authentic - suitable for claims processing")
            else:
                validation_result['warnings'].append("AI detection unavailable - proceeding with standard analysis")
            
            # Quality checks
            width, height = image.size
            if width < 400 or height < 400:
                validation_result['warnings'].append("Low resolution image may affect damage assessment accuracy")
                validation_result['recommendations'].append("Use higher resolution images for better assessment")
            
            if width > 4000 or height > 4000:
                validation_result['warnings'].append("Very high resolution image - processing may take longer")
            
            return validation_result
            
        except Exception as e:
            logger.error(f"Claims image validation error: {e}")
            validation_result['warnings'].append("Image validation incomplete - proceeding with analysis")
            return validation_result

    def detect_with_yolo(self, image):
        """Detect damage using YOLO model for claims assessment"""
        detections = []
        
        # If no model loaded, return demo detections
        if not self.model_loaded:
            logger.info("Using demo damage detection mode")
            # Return some sample detections for demonstration
            return [
                {
                    'type': 'scratch',
                    'confidence': 0.85,
                    'severity': 'low',
                    'severity_multiplier': 1.0,
                    'bbox': {
                        'x': 0.2,
                        'y': 0.3,
                        'width': 0.1,
                        'height': 0.05,
                        'area': 0.005
                    },
                    'cost_estimation': {
                        'min': 100,
                        'max': 300,
                        'base': 200
                    },
                    'repair_priority': 'low'
                },
                {
                    'type': 'dent',
                    'confidence': 0.72,
                    'severity': 'medium',
                    'severity_multiplier': 1.2,
                    'bbox': {
                        'x': 0.5,
                        'y': 0.4,
                        'width': 0.15,
                        'height': 0.12,
                        'area': 0.018
                    },
                    'cost_estimation': {
                        'min': 180,
                        'max': 960,
                        'base': 360
                    },
                    'repair_priority': 'medium'
                }
            ]
        
        try:
            # Run YOLO inference
            results = self.model(image)
            
            for result in results:
                if hasattr(result, 'boxes') and result.boxes is not None:
                    for box in result.boxes:
                        if len(box.data) > 0:
                            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                            confidence = box.conf.item()
                            class_id = int(box.cls.item())
                            
                            # Get class name
                            if class_id < len(self.damage_classes):
                                class_name = self.damage_classes[class_id]
                            else:
                                class_name = f"class_{class_id}"
                            
                            # Convert to normalized coordinates
                            height, width = image.shape[:2]
                            x1_norm = x1 / width
                            y1_norm = y1 / height
                            width_norm = (x2 - x1) / width
                            height_norm = (y2 - y1) / height
                            
                            # Calculate damage severity and cost range
                            damage_info = self.repair_costs.get(class_name, {'min': 200, 'max': 500, 'base': 300, 'severity': 'medium'})
                            severity_multiplier = self.calculate_severity_multiplier(confidence, width_norm * height_norm)
                            
                            estimated_min = damage_info['min'] * severity_multiplier
                            estimated_max = damage_info['max'] * severity_multiplier
                            estimated_base = damage_info['base'] * severity_multiplier
                            
                            detections.append({
                                'type': class_name,
                                'confidence': float(confidence),
                                'severity': damage_info['severity'],
                                'severity_multiplier': severity_multiplier,
                                'bbox': {
                                    'x': float(x1_norm),
                                    'y': float(y1_norm),
                                    'width': float(width_norm),
                                    'height': float(height_norm),
                                    'area': float(width_norm * height_norm)
                                },
                                'cost_estimation': {
                                    'min': estimated_min,
                                    'max': estimated_max,
                                    'base': estimated_base
                                },
                                'repair_priority': self.get_repair_priority(class_name, severity_multiplier)
                            })
            
            logger.info(f"YOLO detection completed: {len(detections)} detections for claims")
            return detections
            
        except Exception as e:
            logger.error(f"YOLO detection failed: {e}")
            # Return empty detections if YOLO fails
            return []

    def calculate_severity_multiplier(self, confidence, area):
        """Calculate severity multiplier based on confidence and damage area"""
        base_multiplier = 1.0
        # Higher confidence and larger area increase severity
        if confidence > 0.8:
            base_multiplier *= 1.3
        elif confidence > 0.6:
            base_multiplier *= 1.1
        
        if area > 0.1:  # Large damage area
            base_multiplier *= 1.5
        elif area > 0.05:  # Medium damage area
            base_multiplier *= 1.2
        
        return min(base_multiplier, 2.5)  # Cap at 2.5x

    def get_repair_priority(self, damage_type, severity_multiplier):
        """Determine repair priority for claims processing"""
        priority_map = {
            'broken_glass': 'urgent',
            'flat_tire': 'urgent',
            'crack': 'high',
            'broken_light': 'medium',
            'dent': 'medium',
            'scratch': 'low'
        }
        
        base_priority = priority_map.get(damage_type, 'medium')
        
        # Adjust priority based on severity
        if severity_multiplier > 1.5:
            if base_priority == 'low':
                return 'medium'
            elif base_priority == 'medium':
                return 'high'
        elif severity_multiplier > 2.0:
            return 'urgent'
        
        return base_priority

    def analyze_damage_for_claims(self, image_data, claim_info=None):
        """Analyze image for insurance claims with AI detection"""
        try:
            cv_image, pil_image = self.preprocess_image(image_data)
            
            # Smart image validation for claims
            validation = self.validate_image_for_claims(pil_image)
            
            # Always proceed with analysis, but include AI detection info
            detections = self.detect_with_yolo(cv_image)
            
            # Create annotated image with claims-specific overlay
            annotated_image = self.draw_detections_with_claims_info(pil_image, detections, validation)
            
            # Prepare comprehensive claims analysis result
            analysis_result = self.prepare_claims_analysis_result(detections, annotated_image, validation, claim_info)
            
            return analysis_result
            
        except Exception as e:
            logger.error(f"Error in claims damage analysis: {e}")
            raise e

    def draw_detections_with_claims_info(self, image, detections, validation):
        """Draw bounding boxes and claims-specific overlay"""
        # Convert to OpenCV for drawing
        cv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        img_height, img_width = cv_image.shape[:2]
        
        # Color scheme for claims
        colors_map = {
            'dent': (255, 152, 0),      # Orange
            'scratch': (255, 193, 7),   # Amber
            'crack': (244, 67, 54),     # Red
            'broken_glass': (33, 150, 243),  # Blue
            'broken_light': (156, 39, 176),  # Purple
            'flat_tire': (121, 85, 72),      # Brown
        }
        
        # Priority colors
        priority_colors = {
            'urgent': (220, 20, 60),    # Crimson
            'high': (255, 140, 0),      # Dark Orange
            'medium': (255, 215, 0),    # Gold
            'low': (50, 205, 50)        # Lime Green
        }
        
        # Draw damage detections
        for detection in detections:
            damage_type = detection['type']
            bbox = detection['bbox']
            confidence = detection['confidence']
            priority = detection['repair_priority']
            
            # Convert normalized coordinates to pixel coordinates
            x = int(bbox['x'] * img_width)
            y = int(bbox['y'] * img_height)
            width = int(bbox['width'] * img_width)
            height = int(bbox['height'] * img_height)
            
            color = colors_map.get(damage_type, (255, 255, 255))
            priority_color = priority_colors.get(priority, (255, 255, 255))
            
            # Draw bounding box
            cv2.rectangle(cv_image, (x, y), (x + width, y + height), color, 3)
            
            # Draw priority indicator
            cv2.rectangle(cv_image, (x, y - 60), (x + 60, y), priority_color, -1)
            cv2.putText(cv_image, priority.upper(), (x + 5, y - 35), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            
            # Draw label
            label = f"{damage_type} ({confidence:.1%})"
            (label_width, label_height), baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
            )
            
            cv2.rectangle(cv_image, (x, y - label_height - 70), 
                         (x + label_width, y - 60), color, -1)
            cv2.putText(cv_image, label, (x, y - 65), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Convert back to PIL
        result_image = Image.fromarray(cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB))
        
        # Add claims-specific overlay
        result_image = self.add_claims_overlay(result_image, validation, len(detections))
        
        return result_image

    def add_claims_overlay(self, image, validation, damage_count):
        """Add claims processing information overlay"""
        draw = ImageDraw.Draw(image)
        width, height = image.size
        
        try:
            font_size = max(20, height // 25)
            title_font = ImageFont.truetype("arial.ttf", font_size)
            normal_font = ImageFont.truetype("arial.ttf", font_size - 4)
        except:
            # Fallback to default font
            try:
                title_font = ImageFont.load_default()
                normal_font = ImageFont.load_default()
            except:
                title_font = normal_font = None
        
        # Claims info box
        box_height = 120
        
        # Draw semi-transparent background
        overlay = Image.new('RGBA', image.size, (0, 0, 0, 0))
        overlay_draw = ImageDraw.Draw(overlay)
        overlay_draw.rectangle([0, 0, width, box_height], fill=(0, 0, 0, 180))
        
        # Composite the overlay
        image = Image.alpha_composite(image.convert('RGBA'), overlay).convert('RGB')
        draw = ImageDraw.Draw(image)
        
        # Claims information text
        claim_id = f"CLAIM-{datetime.now().strftime('%Y%m%d')}-{uuid.uuid4().hex[:6].upper()}"
        
        # Risk level indicator
        risk_level = validation.get('risk_level', 'low')
        risk_colors = {
            'very_high': (220, 20, 60),
            'high': (255, 69, 0),
            'medium': (255, 165, 0),
            'low': (50, 205, 50),
            'verified_real': (0, 128, 0),
            'likely_real': (144, 238, 144)
        }
        
        risk_color = risk_colors.get(risk_level, (255, 255, 255))
        
        # Draw claims info
        y_offset = 10
        if title_font:
            draw.text((20, y_offset), f"CLAIM ID: {claim_id}", fill=(255, 255, 255), font=title_font)
        else:
            draw.text((20, y_offset), f"CLAIM ID: {claim_id}", fill=(255, 255, 255))
        y_offset += 35
        
        risk_text = f"RISK LEVEL: {risk_level.upper().replace('_', ' ')}"
        if normal_font:
            draw.text((20, y_offset), risk_text, fill=risk_color, font=normal_font)
        else:
            draw.text((20, y_offset), risk_text, fill=risk_color)
        y_offset += 25
        
        damage_text = f"DAMAGE COUNT: {damage_count}"
        if normal_font:
            draw.text((20, y_offset), damage_text, fill=(255, 255, 255), font=normal_font)
        else:
            draw.text((20, y_offset), damage_text, fill=(255, 255, 255))
        
        return image

    def prepare_claims_analysis_result(self, detections, annotated_image, validation, claim_info):
        """Prepare comprehensive claims analysis results"""
        total_cost_min = sum(detection['cost_estimation']['min'] for detection in detections)
        total_cost_max = sum(detection['cost_estimation']['max'] for detection in detections)
        total_cost_base = sum(detection['cost_estimation']['base'] for detection in detections)
        
        annotated_image_b64 = self.image_to_base64(annotated_image)
        
        # Calculate claims processing recommendations
        processing_recommendations = self.generate_claims_recommendations(
            detections, total_cost_base, validation
        )
        
        return {
            'claim_id': f"CLAIM-{datetime.now().strftime('%Y%m%d')}-{uuid.uuid4().hex[:6].upper()}",
            'detections': detections,
            'damage_summary': {
                'total_count': len(detections),
                'by_severity': self.summarize_by_severity(detections),
                'by_priority': self.summarize_by_priority(detections),
                'total_cost_range': {
                    'min': total_cost_min,
                    'max': total_cost_max,
                    'base': total_cost_base
                }
            },
            'cost_estimation': {
                'total_usd': total_cost_base,
                'range_usd': {'min': total_cost_min, 'max': total_cost_max},
                'breakdown': [{
                    'type': det['type'],
                    'cost_range': det['cost_estimation'],
                    'priority': det['repair_priority']
                } for det in detections]
            },
            'annotated_image': annotated_image_b64,
            'analysis_date': datetime.now().isoformat(),
            'ai_validation': {
                'is_ai_generated': validation.get('is_ai_generated', False),
                'confidence': validation.get('ai_confidence', 0.0),
                'risk_level': validation.get('risk_level', 'low'),
                'warnings': validation.get('warnings', []),
                'recommendations': validation.get('recommendations', [])
            },
            'claims_processing': {
                'auto_approval_eligible': total_cost_base <= self.claims_thresholds['auto_approve_max'],
                'recommendations': processing_recommendations,
                'estimated_processing_time': self.estimate_processing_time(detections, validation),
                'priority_level': self.determine_claim_priority(detections, validation)
            },
            'success': True
        }

    def generate_claims_recommendations(self, detections, total_cost, validation):
        """Generate claims processing recommendations"""
        recommendations = []
        
        # Cost-based recommendations
        if total_cost == 0:
            recommendations.append("No damage detected - claim may be eligible for quick closure")
        elif total_cost <= self.claims_thresholds['auto_approve_max']:
            recommendations.append("Claim eligible for auto-approval based on cost estimate")
        else:
            recommendations.append("Claim requires manual adjuster review due to cost")
        
        # Risk-based recommendations
        if validation.get('is_ai_generated', False):
            risk_level = validation.get('risk_level', 'medium')
            if risk_level in ['high', 'very_high']:
                recommendations.append("HIGH RISK: Manual image verification required")
            else:
                recommendations.append("Enhanced documentation recommended for AI-detected image")
        
        # Damage-based recommendations
        urgent_damages = [d for d in detections if d['repair_priority'] == 'urgent']
        if urgent_damages:
            recommendations.append("Urgent repairs needed - expedite claim processing")
        
        return recommendations

    def summarize_by_severity(self, detections):
        """Summarize damages by severity"""
        severity_count = {'low': 0, 'medium': 0, 'high': 0}
        for detection in detections:
            severity = detection.get('severity', 'medium')
            severity_count[severity] = severity_count.get(severity, 0) + 1
        return severity_count

    def summarize_by_priority(self, detections):
        """Summarize repairs by priority"""
        priority_count = {'low': 0, 'medium': 0, 'high': 0, 'urgent': 0}
        for detection in detections:
            priority = detection.get('repair_priority', 'medium')
            priority_count[priority] = priority_count.get(priority, 0) + 1
        return priority_count

    def estimate_processing_time(self, detections, validation):
        """Estimate claims processing time"""
        base_time = 2  # days
        
        # Adjust based on damage complexity
        if len(detections) > 3:
            base_time += 1
        if any(d['repair_priority'] == 'urgent' for d in detections):
            base_time -= 1  # Expedite urgent claims
        
        # Adjust based on AI detection risk
        if validation.get('is_ai_generated', False):
            risk_level = validation.get('risk_level', 'medium')
            if risk_level in ['high', 'very_high']:
                base_time += 2  # Additional verification time
        
        return max(1, base_time)  # Minimum 1 day

    def determine_claim_priority(self, detections, validation):
        """Determine overall claim priority"""
        # Check for urgent damages
        if any(d['repair_priority'] == 'urgent' for d in detections):
            return 'urgent'
        
        # Check AI risk level
        if validation.get('is_ai_generated', False):
            risk_level = validation.get('risk_level', 'medium')
            if risk_level in ['high', 'very_high']:
                return 'high'
        
        # Check damage severity
        if any(d['severity'] == 'high' for d in detections):
            return 'high'
        
        return 'normal'

    def image_to_base64(self, image):
        """Convert PIL image to base64"""
        buffered = io.BytesIO()
        image.save(buffered, format="PNG")
        return f"data:image/png;base64,{base64.b64encode(buffered.getvalue()).decode()}"

# Initialize the analyzer
analyzer = ClaimsDamageAnalyzer()

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'service': 'Automated Claims Processing',
        'ai_detection_available': analyzer.ai_detection_available,
        'model_loaded': analyzer.model_loaded,
        'timestamp': datetime.now().isoformat()
    })

@app.route('/api/analyze-damage', methods=['POST'])
def analyze_damage():
    """Main endpoint for claims damage analysis"""
    try:
        data = request.get_json()
        if not data:
            return jsonify({
                'success': False,
                'error': 'No data provided'
            }), 400
        
        image_data = data.get('image')
        if not image_data:
            return jsonify({
                'success': False,
                'error': 'No image data provided'
            }), 400
        
        claim_info = data.get('claim_info', {})
        currency = data.get('currency', 'USD')
        
        # Perform comprehensive damage analysis
        analysis_result = analyzer.analyze_damage_for_claims(image_data, claim_info)
        
        # Convert costs to requested currency
        exchange_rate = analyzer.currency_rates.get(currency, 1.0)
        analysis_result['total_cost'] = analysis_result['cost_estimation']['total_usd'] * exchange_rate
        analysis_result['currency'] = currency
        analysis_result['exchange_rate'] = exchange_rate
        
        logger.info(f"Analysis completed: {analysis_result['damage_summary']['total_count']} detections, AI: {analysis_result['ai_validation']['is_ai_generated']}")
        return jsonify(analysis_result)
        
    except Exception as e:
        logger.error(f"Error in damage analysis: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/generate-report', methods=['POST'])
def generate_report():
    """Generate PDF report for claims"""
    try:
        data = request.get_json()
        if not data or 'analysis_result' not in data:
            return jsonify({
                'success': False,
                'error': 'No analysis result provided'
            }), 400
        
        analysis_result = data['analysis_result']
        currency = data.get('currency', 'USD')
        
        # Generate PDF
        pdf_data = analyzer.pdf_generator.generate_claims_report(analysis_result, currency)
        
        # Create filename
        claim_id = analysis_result.get('claim_id', 'CLAIM-UNKNOWN')
        filename = f"{claim_id}_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
        
        # Return PDF as base64 for frontend download
        pdf_base64 = base64.b64encode(pdf_data).decode('utf-8')
        
        return jsonify({
            'success': True,
            'pdf_data': pdf_base64,
            'filename': filename,
            'message': 'PDF report generated successfully'
        })
        
    except Exception as e:
        logger.error(f"Error generating report: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/validate-image', methods=['POST'])
def validate_image():
    """Standalone image validation endpoint"""
    try:
        data = request.get_json()
        if not data or 'image' not in data:
            return jsonify({
                'success': False,
                'error': 'No image data provided'
            }), 400
        
        cv_image, pil_image = analyzer.preprocess_image(data['image'])
        validation = analyzer.validate_image_for_claims(pil_image)
        
        return jsonify({
            'success': True,
            'validation': validation
        })
        
    except Exception as e:
        logger.error(f"Error in image validation: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

# Also keep the original endpoints for compatibility
@app.route('/api/analyze', methods=['POST'])
def analyze_image_legacy():
    """Legacy endpoint for compatibility"""
    return analyze_damage()

@app.route('/api/generate-pdf', methods=['POST'])
def generate_pdf_legacy():
    """Legacy endpoint for compatibility"""
    return generate_report()

if __name__ == '__main__':
    logger.info("Starting Claims Processing Server...")
    logger.info(f"Model loaded: {analyzer.model_loaded}")
    logger.info(f"AI Detection: {analyzer.ai_detection_available}")
    logger.info("Server running on http://localhost:5000")
    app.run(debug=True, host='0.0.0.0', port=5000)